package main

import (
	"bufio"
	"flag"
	"fmt"
	"internal/pkg/configs"
	"internal/pkg/ssh_util"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"gopkg.in/yaml.v3"
)

const APEX_PATH = "~/rl-research/dqn/ape_x"
const SSH_ARGS = "-oStrictHostKeyChecking=no -oConnectTimeout=5"
const USERNAME = "ehuang"
const FQDN = "cs.mcgill.ca"
const KillPythonProcessesCmd = "pkill \"python\""

func createLearnerCmd(config configs.DistributedConfig) string {
	commands := []string{
		fmt.Sprintf("cd %s", APEX_PATH),
		"conda activate ml",
		fmt.Sprintf("python3 main_learner.py --config_file %s", config.LearnerConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	return cmd
}

func createWorkerCommand(config configs.DistributedConfig, rank int, name string) string {
	worldSize := len(config.ActorHosts) + 3
	commands := []string{
		fmt.Sprintf("cd %s", APEX_PATH),
		"conda activate ml",
		fmt.Sprintf("python3 remote_worker.py --rank %d --name %s --world_size %d --master_addr %s --rpc_port %d --pg_port %d", rank, name, worldSize, config.MasterHost, config.RPCPort, config.PGPort),
	}

	cmd := strings.Join(commands, "; ")
	return cmd
}

func createReplayWorkerCommand(config configs.DistributedConfig) string {
	return createWorkerCommand(config, 1, "replay_server")
}

func createStorageWorkerCommand(config configs.DistributedConfig) string {
	return createWorkerCommand(config, 2, "parameter_server")
}

func createActorWorkerCommand(config configs.DistributedConfig, actorNum int) string {
	name := fmt.Sprintf("actor_%d", actorNum)
	return createWorkerCommand(config, actorNum+3, name)
}

func copyTrainingGraphsToStaticSite(client *ssh_util.Client, learnerName string) {
	cmd := fmt.Sprintf("cp %s/checkpoints/%s/graphs/%s.png ~/public_html/training/learner.png; cp %s/checkpoints/graphs/spectator.png ~/public_html/training/spectator.png\n", APEX_PATH, learnerName, learnerName, APEX_PATH)

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

func runLearner(conf configs.DistributedConfig, learnerName string, SSHUsername string) *ssh_util.Client {
	client := ssh_util.NewClient(conf.LearnerHost, SSHUsername, learnerName)
	defer client.Close()
	cmdSession, err := client.Start(createLearnerCmd(conf), KillPythonProcessesCmd)
	if err != nil {
		panic(err)
	}
	go cmdSession.StreamOutput("[learner] ")
	return client
}

func runWorkers(conf configs.DistributedConfig, learnerName string, SSHUsername string) {
	replayClient := ssh_util.NewClient(conf.ReplayHost, SSHUsername, "replay")
	defer replayClient.Close()
	storageClient := ssh_util.NewClient(conf.StorageHost, SSHUsername, "mongo")
	defer storageClient.Close()

	replayCommandSession, err := replayClient.Start(createReplayWorkerCommand(conf), KillPythonProcessesCmd)
	if err != nil {
		panic(err)
	}
	go replayCommandSession.StreamOutput("[replay] ")
	storageCommandSession, err := storageClient.Start(createStorageWorkerCommand(conf), KillPythonProcessesCmd)
	if err != nil {
		panic(err)
	}
	go storageCommandSession.StreamOutput("[storage] ")

	actorClients := []*ssh_util.Client{}
	for _, host := range conf.ActorHosts {
		client := ssh_util.NewClient(host, SSHUsername, fmt.Sprintf("actor %s", host))
		defer client.Close()
		actorClients = append(actorClients, client)
	}

	for i, client := range actorClients {
		cmd := createActorWorkerCommand(conf, i)
		actorCommandSession, err := client.Start(cmd, KillPythonProcessesCmd)
		if err != nil {
			log.Printf("Warning: %s failed to start: %s\n", client.Name, err)
		}
		go actorCommandSession.StreamOutput(fmt.Sprintf("[%s] ", client.Name))
	}

	updaterClient := ssh_util.NewClient(fmt.Sprintf("mimi.%s", FQDN), SSHUsername, "updator")
	defer updaterClient.Close()
	doneChannel := make(chan bool)
	defer func() {
		doneChannel <- true
		close(doneChannel)
	}()
	runUpdator(updaterClient, 3*time.Second, learnerName, doneChannel)

	cancelChan := make(chan os.Signal, 1)
	signal.Notify(cancelChan, syscall.SIGTERM, syscall.SIGINT)
	sig := <-cancelChan // block here until a sigterm or sigint is recieved

	log.Printf("Caught signal %v", sig)
	fmt.Println("stopping and cleaning up...")
}

func main() {
	f, _ := os.Create("hyperopt_go.log")
	defer f.Close()
	w := bufio.NewWriter(f)
	log.SetOutput(w)

	distributedConfigFilename := flag.String("distributed_config", "generated/distributed_config.yaml", "")
	SSHUsernameFlag := flag.String("ssh_username", USERNAME, "")
	learnerName := flag.String("learner_name", "learner", "")

	flag.Parse()

	distributedConfigFileContent, err := os.ReadFile(*distributedConfigFilename)
	if err != nil {
		panic(err)
	}
	distributedConfig := configs.DistributedConfig{}

	yaml.Unmarshal(distributedConfigFileContent, &distributedConfig)

	log.Printf("%-30s | %+v\n", "Using distributed config: ", distributedConfig)

	if distributedConfig.WithLearner {
		learnerClient := runLearner(distributedConfig, *learnerName, *SSHUsernameFlag)
		defer learnerClient.Close()
	}

	runWorkers(distributedConfig, *learnerName, *SSHUsernameFlag)

	log.Println("go process done")
}
