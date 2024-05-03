package main

import (
	"bufio"
	"flag"
	"fmt"
	"internal/pkg/configs"
	"internal/pkg/ssh_util"
	"math"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"golang.org/x/crypto/ssh"
	"gopkg.in/yaml.v3"
)

const SSH_ARGS = "-oStrictHostKeyChecking=no -oConnectTimeout=5"
const USERNAME = "ehuang"
const FQDN = "cs.mcgill.ca"

func StartReplay(client *ssh_util.Client, config configs.DistributedConfig) *ssh_util.CommandSession {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 distributed_replay_buffer.py --learner_port %d --actors_port %d --config_file %s", config.ReplayLearnerPort, config.ReplayActorsPort, config.ReplayConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func StartMongo(client *ssh_util.Client, config configs.DistributedConfig) *ssh_util.CommandSession {
	commands := []string{
		fmt.Sprintf("mongod --dbpath ~/mongodb/data --logpath ~/mongodb/mongod.log --port %d --auth --bind_ip_all", config.MongoPort),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func StartLearner(client *ssh_util.Client, config configs.DistributedConfig) *ssh_util.CommandSession {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_learner.py --config_file %s", config.LearnerConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func StartActor(client *ssh_util.Client, id string, noisySigma float64, config configs.DistributedConfig) *ssh_util.CommandSession {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --noisy_sigma %.8f", config.ActorConfigFilename, id, noisySigma),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func StartSpectator(client *ssh_util.Client, id string, config configs.DistributedConfig) *ssh_util.CommandSession {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --spectator --noisy_sigma 0", config.ActorConfigFilename, id),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func killMongo(client *ssh.Client) {
	cmd := "mongod --shutdown --dbpath ~/mongodb/data"

	session, err := client.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}

	fmt.Println("Running command: ", cmd)

	if err := session.Run(cmd); err != nil {
		fmt.Println("Failed to kill mongo: ", err)
	}
}

func copyTrainingGraphsToStaticSite(client *ssh_util.Client) {
	cmd := "cp ~/rl-research/ape_x/training_graphs/learner/learner.png ~/public_html/training/learner.png; cp ~/rl-research/ape_x/training_graphs/spectator/spectator.png ~/public_html/training/spectator.png"

	if _, err := client.Run(cmd); err != nil {
		fmt.Println("Failed to copy training graphs: ", err)
	}
}

const baseNoisySigma = 0.9
const alpha = 20

func main_2(distributedConfig configs.DistributedConfig) {
	totalActors := len(distributedConfig.ActorHosts)

	noisySigmas := make([]float64, totalActors)

	for i := 0; i < totalActors; i++ {
		e_i := math.Pow(baseNoisySigma, 1+(float64(i)/float64(totalActors))*alpha)
		noisySigmas[i] = e_i
	}

	replayClient := ssh_util.NewClient(distributedConfig.ReplayHost, USERNAME, "replay")
	mongoClient := ssh_util.NewClient(distributedConfig.MongoHost, USERNAME, "mongo")
	spectatorClient := ssh_util.NewClient(distributedConfig.SpectatorHost, USERNAME, "spectator")

	defer replayClient.Close()
	defer mongoClient.Close()
	defer spectatorClient.Close()

	replayCommandSession := StartReplay(replayClient, distributedConfig)
	mongoCommandSession := StartMongo(mongoClient, distributedConfig)
	spectatorCommandSession := StartSpectator(spectatorClient, "spectator", distributedConfig)

	replayCommandSession.StreamOutput("[replay] ")
	mongoCommandSession.StreamOutput("[mongo] ")
	spectatorCommandSession.StreamOutput("[spectator]")

	actorClients := []*ssh_util.Client{}
	for _, host := range distributedConfig.ActorHosts {
		client := ssh_util.NewClient(host, USERNAME, fmt.Sprintf("actor %s", host))
		defer client.Close()
		actorClients = append(actorClients, client)
	}

	for i, client := range actorClients {
		actorCommandSession := StartActor(client, uuid.New().String(), noisySigmas[i], distributedConfig)
		actorCommandSession.StreamOutput(fmt.Sprintf("[%s] ", client.Name))
	}

	updaterClient := ssh_util.NewClient(fmt.Sprintf("mimi.%s", FQDN), USERNAME, "updator")

	ticker := time.NewTicker(10 * time.Second)
	doneChannel := make(chan bool)
	go func() {
		for {
			select {
			case <-ticker.C:
				copyTrainingGraphsToStaticSite(updaterClient)
			case <-doneChannel:
				ticker.Stop()
				return
			}
		}
	}()

	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n')

	wg := sync.WaitGroup{}

	wg.Add(1)
	go func() {
		replayClient.KillByName("distributed_replay_buffer.py")
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		killMongo(mongoClient.SSHClient)
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		spectatorClient.KillByName("main_actor.py")
		wg.Done()
	}()

	for _, client := range actorClients {
		wg.Add(1)
		go func(c *ssh_util.Client) {
			c.KillByName("main_actor.py")
			wg.Done()
		}(client)
	}

	doneChannel <- true
	close(doneChannel)

	wg.Wait()
}

func main_1(distributedConfig configs.DistributedConfig) {
	totalActors := len(distributedConfig.ActorHosts)
	noisySigmas := make([]float64, totalActors)

	for i := 0; i < totalActors; i++ {
		e_i := math.Pow(baseNoisySigma, 1+(float64(i)/float64(totalActors))*alpha)
		noisySigmas[i] = e_i
	}

	replayClient := ssh_util.NewClient(distributedConfig.ReplayHost, USERNAME, "replay")
	mongoClient := ssh_util.NewClient(distributedConfig.MongoHost, USERNAME, "mongo")
	learnerClient := ssh_util.NewClient(distributedConfig.LearnerHost, USERNAME, "learner")
	spectatorClient := ssh_util.NewClient(distributedConfig.SpectatorHost, USERNAME, "spectator")

	defer replayClient.Close()
	defer mongoClient.Close()
	defer learnerClient.Close()
	defer spectatorClient.Close()

	replayCommandSession := StartReplay(replayClient, distributedConfig)
	mongoCommandSession := StartMongo(mongoClient, distributedConfig)
	learnerCommandSession := StartLearner(learnerClient, distributedConfig)
	spectatorCommandSession := StartSpectator(spectatorClient, "spectator", distributedConfig)

	replayCommandSession.StreamOutput("[replay] ")
	mongoCommandSession.StreamOutput("[mongo] ")
	learnerCommandSession.StreamOutput("[learner] ")
	spectatorCommandSession.StreamOutput("[spectator]")

	actorClients := []*ssh_util.Client{}
	for _, host := range distributedConfig.ActorHosts {
		client := ssh_util.NewClient(host, USERNAME, fmt.Sprintf("actor %s", host))
		defer client.Close()
		actorClients = append(actorClients, client)
	}

	for i, client := range actorClients {
		actorCommandSession := StartActor(client, uuid.New().String(), noisySigmas[i], distributedConfig)
		actorCommandSession.StreamOutput(fmt.Sprintf("[%s] ", client.Name))
	}

	updaterClient := ssh_util.NewClient(fmt.Sprintf("mimi.%s", FQDN), USERNAME, "updator")

	ticker := time.NewTicker(10 * time.Second)
	doneChannel := make(chan bool)
	go func() {
		for {
			select {
			case <-ticker.C:
				copyTrainingGraphsToStaticSite(updaterClient)
			case <-doneChannel:
				ticker.Stop()
				return
			}
		}
	}()

	reader := bufio.NewReader(os.Stdin)
	reader.ReadString('\n')

	wg := sync.WaitGroup{}

	wg.Add(1)
	go func() {
		replayClient.KillByName("distributed_replay_buffer.py")
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		killMongo(mongoClient.SSHClient)
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		learnerClient.KillByName("main_learner.py")
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		spectatorClient.KillByName("main_actor.py")
		wg.Done()
	}()

	for _, client := range actorClients {
		wg.Add(1)
		go func(c *ssh_util.Client) {
			c.KillByName("main_actor.py")
			wg.Done()
		}(client)
	}

	doneChannel <- true
	close(doneChannel)

	wg.Wait()
}

func main() {
	distributedConfigFilename := flag.String("distributed_config", "generated/distributed_config.yaml", "")

	flag.Parse()

	distributedConfigFileContent, err := os.ReadFile(*distributedConfigFilename)
	if err != nil {
		panic(err)
	}
	distributedConfig := configs.DistributedConfig{}

	yaml.Unmarshal(distributedConfigFileContent, &distributedConfig)

	fmt.Printf("%-30s | %+v\n", "Using distributed config: ", distributedConfig)

	if !distributedConfig.WithLearner {
		main_2(distributedConfig)
	} else {
		main_1(distributedConfig)
	}
}
