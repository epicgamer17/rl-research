package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
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
const OutputDir = "generated"
const ReplayLearnerPort = 5554
const ReplayActorPort = 5555
const MongoPort = 5553

func StartReplay(client *ssh_util.Client, replayConfigFilename string) *ssh_util.CommandSession {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 distributed_replay_buffer.py --learner_port %d --actors_port %d --config_file %s", ReplayLearnerPort, ReplayActorPort, replayConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func StartMongo(client *ssh_util.Client, port int) *ssh_util.CommandSession {
	commands := []string{
		fmt.Sprintf("mongod --dbpath ~/mongodb/data --logpath ~/mongodb/mongod.log --port %d --auth --bind_ip_all", port),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func StartLearner(client *ssh_util.Client, learnerConfigFilename string) *ssh_util.CommandSession {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_learner.py --config_file %s", learnerConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func StartActor(client *ssh_util.Client, id string, noisySigma float64, actorConfigFilename string) *ssh_util.CommandSession {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --noisy_sigma %.8f", actorConfigFilename, id, noisySigma),
	}

	cmd := strings.Join(commands, "; ")
	return client.Start(cmd)
}

func StartSpectator(client *ssh_util.Client, id string, actorConfigFilename string) *ssh_util.CommandSession {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --spectator --noisy_sigma 0", actorConfigFilename, id),
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

func main_2(hosts []string) {
	if len(hosts) < 4 {
		panic("Not enough available hosts to run.")
	}

	replayClientHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]
	mongoClientHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]
	spectatorClientHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]

	totalActors := len(hosts)

	fmt.Println("Replay client: ", replayClientHost)
	fmt.Println("Mongo client: ", mongoClientHost)
	fmt.Println("Num actors: ", totalActors)

	noisySigmas := make([]float64, totalActors)

	for i := 0; i < totalActors; i++ {
		e_i := math.Pow(baseNoisySigma, 1+(float64(i)/float64(totalActors))*alpha)
		noisySigmas[i] = e_i
	}

	replayClient := ssh_util.NewClient(replayClientHost, USERNAME, "replay")
	mongoClient := ssh_util.NewClient(mongoClientHost, USERNAME, "mongo")
	spectatorClient := ssh_util.NewClient(spectatorClientHost, USERNAME, "spectator")

	defer replayClient.Close()
	defer mongoClient.Close()
	defer spectatorClient.Close()

	replayCommandSession := StartReplay(replayClient, fmt.Sprintf("%s/replay_config.yaml", OutputDir))
	mongoCommandSession := StartMongo(mongoClient, MongoPort)
	spectatorCommandSession := StartSpectator(spectatorClient, "spectator", fmt.Sprintf("%s/actor_config.yaml", OutputDir))

	replayCommandSession.StreamOutput("[replay] ")
	mongoCommandSession.StreamOutput("[mongo] ")
	spectatorCommandSession.StreamOutput("[spectator]")

	actorClients := []*ssh_util.Client{}
	for _, host := range hosts {
		client := ssh_util.NewClient(host, USERNAME, fmt.Sprintf("actor %s", host))
		defer client.Close()
		actorClients = append(actorClients, client)
	}

	for i, client := range actorClients {
		actorCommandSession := StartActor(client, uuid.New().String(), noisySigmas[i], fmt.Sprintf("%s/actor_config.yaml", OutputDir))
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

func main_1(hosts []string) {
	if len(hosts) < 5 {
		panic("Not enough available hosts to run.")
	}

	replayClientHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]
	mongoClientHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]
	learnerClientHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]
	spectatorClientHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]

	totalActors := len(hosts)

	fmt.Println("Replay client: ", replayClientHost)
	fmt.Println("Mongo client: ", mongoClientHost)
	fmt.Println("Learner client: ", learnerClientHost)
	fmt.Println("Num actors: ", totalActors)

	noisySigmas := make([]float64, totalActors)

	for i := 0; i < totalActors; i++ {
		e_i := math.Pow(baseNoisySigma, 1+(float64(i)/float64(totalActors))*alpha)
		noisySigmas[i] = e_i
	}

	replayClient := ssh_util.NewClient(replayClientHost, USERNAME, "replay")
	mongoClient := ssh_util.NewClient(mongoClientHost, USERNAME, "mongo")
	learnerClient := ssh_util.NewClient(learnerClientHost, USERNAME, "learner")
	spectatorClient := ssh_util.NewClient(spectatorClientHost, USERNAME, "spectator")

	defer replayClient.Close()
	defer mongoClient.Close()
	defer learnerClient.Close()
	defer spectatorClient.Close()

	replayCommandSession := StartReplay(replayClient, fmt.Sprintf("%s/replay_config.yaml", OutputDir))
	mongoCommandSession := StartMongo(mongoClient, MongoPort)
	learnerCommandSession := StartLearner(learnerClient, fmt.Sprintf("%s/learner_config.yaml", OutputDir))
	spectatorCommandSession := StartSpectator(spectatorClient, "spectator", fmt.Sprintf("%s/actor_config.yaml", OutputDir))

	replayCommandSession.StreamOutput("[replay] ")
	mongoCommandSession.StreamOutput("[mongo] ")
	learnerCommandSession.StreamOutput("[learner] ")
	spectatorCommandSession.StreamOutput("[spectator]")

	actorClients := []*ssh_util.Client{}
	for _, host := range hosts {
		client := ssh_util.NewClient(host, USERNAME, fmt.Sprintf("actor %s", host))
		defer client.Close()
		actorClients = append(actorClients, client)
	}

	for i, client := range actorClients {
		actorCommandSession := StartActor(client, uuid.New().String(), noisySigmas[i], fmt.Sprintf("%s/actor_config.yaml", OutputDir))
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
	learnerBaseFilenameFlag := flag.String("learner_config", "../configs/actor_config_example.yaml", "")
	actorBaseFilenameFlag := flag.String("actor_config", "../configs/learner_config_example.yaml", "")
	replayFilenameFlag := flag.String("replay_config_file", "../configs/replay_config_example.yaml", "")
	hostsFilenameFlag := flag.String("hosts_file", "../generated/hosts.yaml", "")
	hyperoptModeFlag := flag.Bool("hyperopt", false, "")
	flag.Parse()

	fmt.Printf("%-30s | %v\n", "Using learner base config file", *learnerBaseFilenameFlag)
	fmt.Printf("%-30s | %v\n", "Using actor base config file", *actorBaseFilenameFlag)
	fmt.Printf("%-30s | %v\n", "Using replay config file", *replayFilenameFlag)
	fmt.Printf("%-30s | %v\n", "Using hyperopt mode", *hyperoptModeFlag)

	hostsFileContent, err := os.ReadFile(*hostsFilenameFlag)
	if err != nil {
		panic(err)
	}
	hosts := []string{}

	yaml.Unmarshal(hostsFileContent, &hosts)

	fmt.Printf("%-30s | %v\n", "Using hosts", hosts)

	if _, err := os.Stat(*learnerBaseFilenameFlag); errors.Is(err, os.ErrNotExist) {
		panic("Base rainbow config file does not exist")
	}
	if _, err := os.Stat(*replayFilenameFlag); errors.Is(err, os.ErrNotExist) {
		panic("Replay config file does not exist")
	}

	if *hyperoptModeFlag {
		main_2(hosts)
	} else {
		main_1(hosts)
	}
}
