package main

import (
	"bufio"
	"errors"
	"flag"
	"fmt"
	"io"
	"math"
	"net"
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
const MongoPort = 5553
const ReplayLearnerPort = 5554
const ReplayActorPort = 5555
const MongoUsername = "ezra"
const MongoPasswordLocation = "~/mongodb/mongodb_admin_password"
const OutputDir = "generated"

func NewSSHConfig(username string) *ssh.ClientConfig {
	path := os.ExpandEnv("$HOME/.ssh/id_ed25519")
	bytes, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}

	key, err := ssh.ParsePrivateKey(bytes)
	if err != nil {
		panic(err)
	}

	config := &ssh.ClientConfig{
		User: username,
		Auth: []ssh.AuthMethod{
			ssh.PublicKeys(key),
		},
		HostKeyCallback: ssh.InsecureIgnoreHostKey(),
		Timeout:         15 * time.Second,
	}

	return config
}

func Reader(reader io.Reader) <-chan string {
	ch := make(chan string)
	scanner := bufio.NewScanner(reader)
	go func() {
		for {
			if scanner.Scan() {
				ch <- scanner.Text()
			} else {
				if err := scanner.Err(); err != nil {
					fmt.Println("Error reading from SSH: ", err)
				} else {
					fmt.Println("EOF")
					close(ch)
					break
				}
			}
		}
	}()
	return ch
}

func merge(channels ...<-chan string) <-chan string {
	wg := sync.WaitGroup{}
	wg.Add(len(channels))
	ch := make(chan string)

	output := func(c <-chan string) {
		for n := range c {
			ch <- n
		}
		wg.Done()
	}

	for _, c := range channels {
		go output(c)
	}

	go func() {
		wg.Wait()
		close(ch)
	}()

	return ch
}

func NewClient(host string) *ssh.Client {
	config := NewSSHConfig(USERNAME)
	address := net.JoinHostPort(host, "22")
	client, err := ssh.Dial("tcp", address, config)
	if err != nil {
		panic("failed to dial: " + err.Error())
	}

	return client
}

func createChannels(session *ssh.Session) (<-chan string, <-chan string) {
	stderrPipe, err := session.StderrPipe()
	if err != nil {
		panic("failed to create stderr pipe: " + err.Error())
	}
	stdoutPipe, err := session.StdoutPipe()
	if err != nil {
		panic("failed to create stdout pipe: " + err.Error())
	}

	stdoutCh := Reader(stdoutPipe)
	stderrCh := Reader(stderrPipe)

	return stdoutCh, stderrCh
}

type DistributedConfig struct {
	ReplayHost            string
	ReplayLearnerPort     int
	ReplayActorsPort      int
	MongoHost             string
	MongoPort             int
	MongoUsername         string
	MongoPasswordLocation string
}

func CreateConfigs(client *ssh.Client, config DistributedConfig, learnerConfigFilename string, actorConfigFilename string, replayConfigFilename string) {
	learnerConfig, err := os.ReadFile(learnerConfigFilename)
	if err != nil {
		panic(err)
	}
	actorConfig, err := os.ReadFile(actorConfigFilename)
	if err != nil {
		panic(err)
	}
	replayConfig, err := os.ReadFile(replayConfigFilename)
	if err != nil {
		panic(err)
	}

	fmt.Println("Creating configs: ")
	session, err := client.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}
	defer session.Close()

	// write config files to remote, then run the config_generator.py script to inject the necessary info
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("echo '%s' > \"%s\"", string(learnerConfig), learnerConfigFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(actorConfig), actorConfigFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(replayConfig), replayConfigFilename),
		fmt.Sprintf("python3 config_generator.py --replay_addr %s --replay_learner_port %d --replay_actors_port %d --storage_hostname %s --storage_port %d --storage_username %s --actor_base %s --learner_base %s --output %s", config.ReplayHost, config.ReplayLearnerPort, config.ReplayActorsPort, config.MongoHost, config.MongoPort, config.MongoUsername, actorConfigFilename, learnerConfigFilename, OutputDir),
	}

	cmd := strings.Join(commands, "; ")
	_, err = session.Output(cmd)
	if err != nil {
		fmt.Println(err)
		panic(err)
	}
}

func StartReplay(session *ssh.Session, replayConfigFilename string) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 distributed_replay_buffer.py --learner_port %d --actors_port %d --config_file %s", ReplayLearnerPort, ReplayActorPort, replayConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	session.Start(cmd)
	return stdoutCh, stderrCh
}

func StartMongo(session *ssh.Session, port int) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		fmt.Sprintf("mongod --dbpath ~/mongodb/data --logpath ~/mongodb/mongod.log --port %d --auth --bind_ip_all", port),
	}

	cmd := strings.Join(commands, "; ")
	session.Start(cmd)
	return stdoutCh, stderrCh
}

func StartLearner(session *ssh.Session, learnerConfigFilename string) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_learner.py --config_file %s", learnerConfigFilename),
	}

	cmd := strings.Join(commands, "; ")
	session.Start(cmd)
	return stdoutCh, stderrCh
}

func StartActor(session *ssh.Session, id string, noisySigma float64, actorConfigFilename string) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --noisy_sigma %.8f", actorConfigFilename, id, noisySigma),
	}

	cmd := strings.Join(commands, "; ")
	session.Start(cmd)
	return stdoutCh, stderrCh
}

func StartSpectator(session *ssh.Session, id string, actorConfigFilename string) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --spectator --noisy_sigma 0", actorConfigFilename, id),
	}

	cmd := strings.Join(commands, "; ")
	session.Start(cmd)
	return stdoutCh, stderrCh
}

func killByName(client *ssh.Client, name string) {
	cmd := fmt.Sprintf("kill -9 $(ps aux | grep %s | grep -v grep | awk '{print $2}')", name)

	session, err := client.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}

	fmt.Println("Running command: ", cmd)

	if err := session.Run(cmd); err != nil {
		fmt.Println("Failed to kill ", name, ": ", err)
	}
}

func killReplay(client *ssh.Client) {
	killByName(client, "distributed_replay_buffer.py")
}

func killLearner(client *ssh.Client) {
	killByName(client, "main_learner.py")
}

func killActor(client *ssh.Client) {
	killByName(client, "main_actor.py")
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

func copyTrainingGraphsToStaticSite(client *ssh.Client) {
	cmd := "cp ~/rl-research/ape_x/training_graphs/learner/learner.png ~/public_html/training/learner.png; cp ~/rl-research/ape_x/training_graphs/spectator/spectator.png ~/public_html/training/spectator.png"
	session, err := client.NewSession()
	if err != nil {
		fmt.Println("failed to create session: ", err)
	}
	defer session.Close()

	fmt.Println("Running command: ", cmd)

	if err := session.Run(cmd); err != nil {
		fmt.Println("Failed to copy training graphs: ", err)
	}

}

const baseNoisySigma = 0.9
const alpha = 20

func main_2(hosts []string, learnerBaseFilenameFlag string, actorBaseFilenameFlag string, replayFilenameFlag string) {
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

	replayClient := NewClient(replayClientHost)
	mongoClient := NewClient(mongoClientHost)
	spectatorClient := NewClient(spectatorClientHost)

	defer replayClient.Close()
	defer mongoClient.Close()
	defer spectatorClient.Close()

	distributedConfig := &DistributedConfig{
		ReplayHost:            replayClientHost,
		ReplayLearnerPort:     ReplayLearnerPort,
		ReplayActorsPort:      ReplayActorPort,
		MongoHost:             mongoClientHost,
		MongoPort:             MongoPort,
		MongoUsername:         MongoUsername,
		MongoPasswordLocation: MongoPasswordLocation,
	}

	CreateConfigs(spectatorClient, *distributedConfig, learnerBaseFilenameFlag, actorBaseFilenameFlag, replayFilenameFlag)

	replaySession, err := replayClient.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}
	defer replaySession.Close()

	mongoSession, err := mongoClient.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}
	defer mongoSession.Close()

	spectatorSession, err := spectatorClient.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}
	defer spectatorSession.Close()

	replayStdout, replayStderr := StartReplay(replaySession, replayFilenameFlag)
	mongoStdout, mongoStderr := StartMongo(mongoSession, MongoPort)
	spectatorStdout, spectatorStderr := StartSpectator(spectatorSession, "spectator", fmt.Sprintf("%s/actor_config.yaml", OutputDir))

	go func() {
		for msg := range merge(replayStdout, replayStderr) {
			fmt.Printf("[replay] %s\n", msg)
		}
	}()
	go func() {
		for msg := range merge(mongoStdout, mongoStderr) {
			fmt.Printf("[mongo] %s\n", msg)
		}
	}()
	go func() {
		for msg := range merge(spectatorStdout, spectatorStderr) {
			fmt.Printf("[spectator] %s\n", msg)
		}
	}()

	actorClients := []*ssh.Client{}
	for _, host := range hosts {
		actorClients = append(actorClients, NewClient(host))
	}

	for i, client := range actorClients {
		defer client.Close()

		session, err := client.NewSession()
		if err != nil {
			panic("failed to create session: " + err.Error())
		}
		defer session.Close()

		stdout, stderr := StartActor(session, uuid.New().String(), noisySigmas[i], fmt.Sprintf("%s/actor_config.yaml", OutputDir))

		go func(client *ssh.Client) {
			for msg := range merge(stdout, stderr) {
				fmt.Printf("[actor %s] %s\n", client.RemoteAddr().String(), msg)
			}
		}(client)
	}

	updaterClient := NewClient(fmt.Sprintf("mimi.%s", FQDN))

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
		killReplay(replayClient)
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		killMongo(mongoClient)
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		killActor(spectatorClient)
		wg.Done()
	}()

	for _, client := range actorClients {
		wg.Add(1)
		go func(c *ssh.Client) {
			killActor(c)
			wg.Done()
		}(client)
	}

	doneChannel <- true
	close(doneChannel)

	wg.Wait()
}

func main_1(hosts []string, learnerBaseFilenameFlag string, actorBaseFilenameFlag string, replayFilenameFlag string) {
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

	replayClient := NewClient(replayClientHost)
	mongoClient := NewClient(mongoClientHost)
	learnerClient := NewClient(learnerClientHost)
	spectatorClient := NewClient(spectatorClientHost)

	defer replayClient.Close()
	defer mongoClient.Close()
	defer learnerClient.Close()
	defer spectatorClient.Close()

	distributedConfig := &DistributedConfig{
		ReplayHost:            replayClientHost,
		ReplayLearnerPort:     ReplayLearnerPort,
		ReplayActorsPort:      ReplayActorPort,
		MongoHost:             mongoClientHost,
		MongoPort:             MongoPort,
		MongoUsername:         MongoUsername,
		MongoPasswordLocation: MongoPasswordLocation,
	}

	CreateConfigs(spectatorClient, *distributedConfig, learnerBaseFilenameFlag, actorBaseFilenameFlag, replayFilenameFlag)

	replaySession, err := replayClient.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}
	defer replaySession.Close()

	mongoSession, err := mongoClient.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}
	defer mongoSession.Close()

	learnerSession, err := learnerClient.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}
	defer learnerSession.Close()

	spectatorSession, err := spectatorClient.NewSession()
	if err != nil {
		panic("failed to create session: " + err.Error())
	}
	defer spectatorSession.Close()

	replayStdout, replayStderr := StartReplay(replaySession, replayFilenameFlag)
	mongoStdout, mongoStderr := StartMongo(mongoSession, MongoPort)

	learnerStdout, learnerStderr := StartLearner(learnerSession, fmt.Sprintf("%s/learner_config.yaml", OutputDir))

	spectatorStdout, spectatorStderr := StartSpectator(spectatorSession, "spectator", fmt.Sprintf("%s/actor_config.yaml", OutputDir))

	go func() {
		for msg := range merge(replayStdout, replayStderr) {
			fmt.Printf("[replay] %s\n", msg)
		}
	}()
	go func() {
		for msg := range merge(mongoStdout, mongoStderr) {
			fmt.Printf("[mongo] %s\n", msg)
		}
	}()
	go func() {
		for msg := range merge(learnerStdout, learnerStderr) {
			fmt.Printf("[learner] %s\n", msg)
		}
	}()
	go func() {
		for msg := range merge(spectatorStdout, spectatorStderr) {
			fmt.Printf("[spectator] %s\n", msg)
		}
	}()

	actorClients := []*ssh.Client{}
	for _, host := range hosts {
		actorClients = append(actorClients, NewClient(host))
	}

	for i, client := range actorClients {
		defer client.Close()

		session, err := client.NewSession()
		if err != nil {
			panic("failed to create session: " + err.Error())
		}
		defer session.Close()

		stdout, stderr := StartActor(session, uuid.New().String(), noisySigmas[i], fmt.Sprintf("%s/actor_config.yaml", OutputDir))

		go func(client *ssh.Client) {
			for msg := range merge(stdout, stderr) {
				fmt.Printf("[actor %s] %s\n", client.RemoteAddr().String(), msg)
			}
		}(client)
	}

	updaterClient := NewClient(fmt.Sprintf("mimi.%s", FQDN))

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
		killReplay(replayClient)
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		killMongo(mongoClient)
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		killLearner(learnerClient)
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		killActor(spectatorClient)
		wg.Done()
	}()

	for _, client := range actorClients {
		wg.Add(1)
		go func(c *ssh.Client) {
			killActor(c)
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
		main_2(hosts, *learnerBaseFilenameFlag, *actorBaseFilenameFlag, *replayFilenameFlag)
	} else {
		main_1(hosts, *learnerBaseFilenameFlag, *actorBaseFilenameFlag, *replayFilenameFlag)
	}
}
