package main

import (
	"bufio"
	"flag"
	"fmt"
	"internal/pkg/configs"
	"internal/pkg/ssh_util"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"
	"gopkg.in/yaml.v3"
)

const SSH_ARGS = "-oStrictHostKeyChecking=no -oConnectTimeout=5"
const USERNAME = "ehuang"
const FQDN = "cs.mcgill.ca"
const KillPythonProcessesCmd = "pkill \"python\""

func createReplayCommand(config configs.DistributedConfig) string {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 distributed_replay_buffer.py --learner_port %d --actors_port %d --config_file %s", config.ReplayLearnerPort, config.ReplayActorsPort, config.ReplayConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	return cmd
}

func createMongoCmd(config configs.DistributedConfig) string {
	commands := []string{
		fmt.Sprintf("mongod --dbpath ~/mongodb/data --logpath ~/mongodb/mongod.log --port %d --auth --bind_ip_all", config.MongoPort),
	}

	cmd := strings.Join(commands, "; ")
	return cmd
}

func createLearnerCmd(config configs.DistributedConfig) string {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_learner.py --config_file %s", config.LearnerConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	return cmd
}

func createActorCmd(config configs.DistributedConfig, id string, noisySigma string) string {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --noisy_sigma %s", config.ActorConfigFilename, id, noisySigma),
	}

	cmd := strings.Join(commands, "; ")
	return cmd
}

func createSpectatorCmd(config configs.DistributedConfig, id string) string {
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --spectator --noisy_sigma 0", config.ActorConfigFilename, id),
	}

	cmd := strings.Join(commands, "; ")
	return cmd
}

func copyTrainingGraphsToStaticSite(client *ssh_util.Client, learnerName string) {
	cmd := fmt.Sprintf("cp ~/rl-research/ape_x/training_graphs/%s/%s.png ~/public_html/training/learner.png; cp ~/rl-research/ape_x/training_graphs/spectator/spectator.png ~/public_html/training/spectator.png\n", learnerName, learnerName)

	if _, err := client.Run(cmd); err != nil {
		log.Println("Failed to copy training graphs: ", err)
	}
}

func runUpdator(client *ssh_util.Client, duration time.Duration, learnerName string, doneChannel <-chan bool) {
	ticker := time.NewTicker(duration)
	go func() {
		for {
			select {
			case <-ticker.C:
				copyTrainingGraphsToStaticSite(client, learnerName)
			case <-doneChannel:
				ticker.Stop()
				return
			}
		}
	}()
}

const baseNoisySigma = 0.9
const alpha = 20

func main_2(distributedConfig configs.DistributedConfig, learnerName string) {
	totalActors := len(distributedConfig.ActorHosts)

	noisySigmas := make([]float64, totalActors)

	for i := 0; i < totalActors; i++ {
		e_i := math.Pow(baseNoisySigma, 1+(float64(i)/float64(totalActors))*alpha)
		noisySigmas[i] = e_i
	}

	replayClient := ssh_util.NewClient(distributedConfig.ReplayHost, USERNAME, "replay")
	defer replayClient.Close()
	mongoClient := ssh_util.NewClient(distributedConfig.MongoHost, USERNAME, "mongo")
	defer mongoClient.Close()
	spectatorClient := ssh_util.NewClient(distributedConfig.SpectatorHost, USERNAME, "spectator")
	defer spectatorClient.Close()

	replayCommandSession, err := replayClient.Start(createReplayCommand(distributedConfig), KillPythonProcessesCmd)
	if err != nil {
		panic(err)
	}
	go replayCommandSession.StreamOutput("[replay] ")
	mongoCommandSession, err := mongoClient.Start(createMongoCmd(distributedConfig), "mongod --shutdown --dbpath ~/mongodb/data")
	if err != nil {
		panic(err)
	}
	go mongoCommandSession.StreamOutput("[mongo] ")
	spectatorCommandSession, err := spectatorClient.Start(createSpectatorCmd(distributedConfig, "spectator"), KillPythonProcessesCmd)
	if err != nil {
		log.Println("Warning: spectator failed to start", err)
	}
	go spectatorCommandSession.StreamOutput("[spectator]")

	actorClients := []*ssh_util.Client{}
	for _, host := range distributedConfig.ActorHosts {
		client := ssh_util.NewClient(host, USERNAME, fmt.Sprintf("actor %s", host))
		defer client.Close()
		actorClients = append(actorClients, client)
	}

	for i, client := range actorClients {
		cmd := createActorCmd(distributedConfig, uuid.New().String(), strconv.FormatFloat(noisySigmas[i], 'f', 8, 64))
		actorCommandSession, err := client.Start(cmd, KillPythonProcessesCmd)
		if err != nil {
			log.Printf("Warning: %s failed to start: %s\n", client.Name, err)
		}
		go actorCommandSession.StreamOutput(fmt.Sprintf("[%s] ", client.Name))
	}

	updaterClient := ssh_util.NewClient(fmt.Sprintf("mimi.%s", FQDN), USERNAME, "updator")
	defer updaterClient.Close()
	doneChannel := make(chan bool)
	defer func() {
		doneChannel <- true
		close(doneChannel)
	}()
	runUpdator(updaterClient, 10*time.Second, learnerName, doneChannel)

	reader := bufio.NewReader(os.Stdin)
	_, err = reader.ReadString('\n')
	if err != nil {
		log.Println("error reading string", err)
	}
	log.Println("recieved stop signal, stopping...")
}

func main_1(distributedConfig configs.DistributedConfig, learnerName string) {
	totalActors := len(distributedConfig.ActorHosts)
	noisySigmas := make([]float64, totalActors)

	for i := 0; i < totalActors; i++ {
		e_i := math.Pow(baseNoisySigma, 1+(float64(i)/float64(totalActors))*alpha)
		noisySigmas[i] = e_i
	}

	replayClient := ssh_util.NewClient(distributedConfig.ReplayHost, USERNAME, "replay")
	defer replayClient.Close()
	mongoClient := ssh_util.NewClient(distributedConfig.MongoHost, USERNAME, "mongo")
	defer mongoClient.Close()
	learnerClient := ssh_util.NewClient(distributedConfig.LearnerHost, USERNAME, "learner")
	defer learnerClient.Close()
	spectatorClient := ssh_util.NewClient(distributedConfig.SpectatorHost, USERNAME, "spectator")
	defer spectatorClient.Close()

	replayCommandSession, err := replayClient.Start(createReplayCommand(distributedConfig), KillPythonProcessesCmd)
	if err != nil {
		panic(err)
	}
	go replayCommandSession.StreamOutput("[replay] ")
	mongoCommandSession, err := mongoClient.Start(createMongoCmd(distributedConfig), "mongod --shutdown --dbpath ~/mongodb/data")
	if err != nil {
		panic(err)
	}
	go mongoCommandSession.StreamOutput("[mongo] ")
	spectatorCommandSession, err := spectatorClient.Start(createSpectatorCmd(distributedConfig, "spectator"), KillPythonProcessesCmd)
	if err != nil {
		log.Println("Warning: spectator failed to start", err)
	}
	go spectatorCommandSession.StreamOutput("[spectator]")
	learnerCommandSession, err := learnerClient.Start(createLearnerCmd(distributedConfig), KillPythonProcessesCmd)
	if err != nil {
		panic(err)
	}
	go learnerCommandSession.StreamOutput("[learner] ")

	actorClients := []*ssh_util.Client{}
	for _, host := range distributedConfig.ActorHosts {
		client := ssh_util.NewClient(host, USERNAME, fmt.Sprintf("actor %s", host))
		defer client.Close()
		actorClients = append(actorClients, client)
	}

	for i, client := range actorClients {
		cmd := createActorCmd(distributedConfig, uuid.New().String(), strconv.FormatFloat(noisySigmas[i], 'f', 8, 64))
		actorCommandSession, err := client.Start(cmd, KillPythonProcessesCmd)
		if err != nil {
			log.Printf("Warning: %s failed to start: %s\n", client.Name, err)
		}
		go actorCommandSession.StreamOutput(fmt.Sprintf("[%s] ", client.Name))
	}

	updaterClient := ssh_util.NewClient(fmt.Sprintf("mimi.%s", FQDN), USERNAME, "updator")
	defer updaterClient.Close()
	doneChannel := make(chan bool)
	defer func() {
		doneChannel <- true
		close(doneChannel)
	}()
	runUpdator(updaterClient, 10*time.Second, learnerName, doneChannel)

	reader := bufio.NewReader(os.Stdin)
	_, err = reader.ReadString('\n')
	if err != nil {
		log.Println("error reading string", err)
	}
	log.Println("recieved stop signal, stopping...")
}

func main() {
	f, _ := os.Create("hyperopt_go.log")
	defer f.Close()
	w := bufio.NewWriter(f)
	log.SetOutput(w)

	distributedConfigFilename := flag.String("distributed_config", "generated/distributed_config.yaml", "")
	learnerName := flag.String("learner_name", "learner", "")

	flag.Parse()

	distributedConfigFileContent, err := os.ReadFile(*distributedConfigFilename)
	if err != nil {
		panic(err)
	}
	distributedConfig := configs.DistributedConfig{}

	yaml.Unmarshal(distributedConfigFileContent, &distributedConfig)

	log.Printf("%-30s | %+v\n", "Using distributed config: ", distributedConfig)

	if !distributedConfig.WithLearner {
		main_2(distributedConfig, *learnerName)
	} else {
		main_1(distributedConfig, *learnerName)
	}

	log.Println("go process done")
}
