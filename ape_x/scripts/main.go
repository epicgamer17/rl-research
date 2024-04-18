package main

import (
	"bufio"
	"fmt"
	"io"
	"math"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/fatih/color"
	"github.com/google/uuid"
	"golang.org/x/crypto/ssh"
)

const SSH_ARGS = "-oStrictHostKeyChecking=no -oConnectTimeout=5"
const USERNAME = "ehuang"
const FQDN = "cs.mcgill.ca"
const MongoPort = 5553
const ReplayLearnerPort = 5554
const ReplayActorPort = 5555
const MongoUsername = "ezra"
const MongoPasswordLocation = "~/mongodb/mongodb_admin_password"
const ActorConfigFile = "actor_config_example.yaml"
const LearnerConfigFile = "learner_config_example.yaml"
const ReplayConfigFile = "replay_config_example.yaml"

func GenerateHosts(username string) <-chan string {
	c := make(chan string)
	go func() {
		for i := 1; i <= 33; i++ {
			if i == 17 {
				continue
			}

			host := fmt.Sprintf("open-gpu-%d.%s", i, FQDN)
			c <- host
		}
		close(c)
	}()
	return c
}

func FindFreeServers(hosts <-chan string) (<-chan string, <-chan bool) {
	wg := sync.WaitGroup{}
	c := make(chan string, 100)
	done := make(chan bool)

	for host := range hosts {
		wg.Add(1)
		connectionString := fmt.Sprintf("%s@%s", USERNAME, host)
		// fmt.Println("checking", connectionString)
		args := append(strings.Split(SSH_ARGS, " "), connectionString, "who | cut -d \" \" -f1 | sort -n | uniq -c | wc -l")
		cmd := exec.Command("ssh", args...)

		go func(h string) {
			defer fmt.Printf("%v done", h)
			defer wg.Done()
			output, err := cmd.Output()
			if err == nil {
				o := strings.Trim(string(output), "\n")
				numUsersOnline, err := strconv.Atoi(o)
				if err != nil {
					panic(err)
				}
				if numUsersOnline <= 1 {
					color.Green("  %s is free", h)
					c <- h
				} else {
					color.Yellow("  %s has other users online", h)
				}
			} else {
				color.Red("  %s is not online", h)
			}
		}(host)
	}

	go func() {
		wg.Wait()
		fmt.Println("done")
		done <- true
		close(c)
		close(done)
	}()

	return c, done
}

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

func StartReplay(session *ssh.Session) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("python3 distributed_replay_buffer.py --learner_port %d --actors_port %d --config_file %s", ReplayLearnerPort, ReplayActorPort, ReplayConfigFile),
	}

	cmd := strings.Join(commands, "; ")
	session.Start(cmd)
	return stdoutCh, stderrCh
}

type DistributedConfig struct {
	ReplayHost            string
	ReplayPort            int
	MongoHost             string
	MongoPort             int
	MongoUsername         string
	MongoPasswordLocation string
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

func StartLearner(session *ssh.Session, conf *DistributedConfig) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		// fmt.Sprintf("python3 main_learner.py %s %d %s %d %s $(cat %s)", conf.ReplayHost, conf.ReplayPort, conf.MongoHost, conf.MongoPort, conf.MongoUsername, conf.MongoPasswordLocation),
		fmt.Sprintf("python3 main_learner.py --config_file %s", LearnerConfigFile),
	}

	cmd := strings.Join(commands, "; ")
	session.Start(cmd)
	return stdoutCh, stderrCh
}

func StartActor(session *ssh.Session, conf *DistributedConfig, id string, epsilon float64) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		// fmt.Sprintf("python3 main_actor.py %s %.8f %s %d %s %d %s $(cat %s)", id, epsilon, conf.ReplayHost, conf.ReplayPort, conf.MongoHost, conf.MongoPort, conf.MongoUsername, conf.MongoPasswordLocation),
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s", ActorConfigFile, id),
	}

	cmd := strings.Join(commands, "; ")
	session.Start(cmd)
	return stdoutCh, stderrCh
}

func StartSpectator(session *ssh.Session, conf *DistributedConfig, id string, epsilon float64) (<-chan string, <-chan string) {
	stdoutCh, stderrCh := createChannels(session)
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		// fmt.Sprintf("python3 main_actor.py %s %.8f %s %d %s %d %s $(cat %s) --spectator", id, epsilon, conf.ReplayHost, conf.ReplayPort, conf.MongoHost, conf.MongoPort, conf.MongoUsername, conf.MongoPasswordLocation),
		fmt.Sprintf("python3 main_actor.py --config_file %s --name %s --spectator", ActorConfigFile, id),
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

const baseEpsilon = 0.4

func main() {
	availableHosts, done := FindFreeServers(GenerateHosts(USERNAME))
	<-done

	replayClientHost := <-availableHosts
	mongoClientHost := <-availableHosts
	learnerClientHost := <-availableHosts
	spectatorClientHost := <-availableHosts

	totalActors := len(availableHosts)

	fmt.Println("Replay client: ", replayClientHost)
	fmt.Println("Mongo client: ", mongoClientHost)
	fmt.Println("Learner client: ", learnerClientHost)
	fmt.Println("Num actors: ", totalActors)

	epsilons := make([]float64, totalActors)

	for i := 0; i < totalActors; i++ {
		e_i := math.Pow(baseEpsilon, 1+(float64(i)/float64(totalActors))*7)
		epsilons[i] = e_i
	}

	replayClient := NewClient(replayClientHost)
	mongoClient := NewClient(mongoClientHost)
	learnerClient := NewClient(learnerClientHost)
	spectatorClient := NewClient(spectatorClientHost)

	defer replayClient.Close()
	defer mongoClient.Close()
	defer learnerClient.Close()
	defer spectatorClient.Close()

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

	replayStdout, replayStderr := StartReplay(replaySession)
	mongoStdout, mongoStderr := StartMongo(mongoSession, MongoPort)

	learnerDistributedConfig := &DistributedConfig{
		ReplayHost:            replayClientHost,
		ReplayPort:            ReplayLearnerPort,
		MongoHost:             mongoClientHost,
		MongoPort:             MongoPort,
		MongoUsername:         MongoUsername,
		MongoPasswordLocation: MongoPasswordLocation,
	}

	actorDistributedConfig := &DistributedConfig{
		ReplayHost:            replayClientHost,
		ReplayPort:            ReplayActorPort,
		MongoHost:             mongoClientHost,
		MongoPort:             MongoPort,
		MongoUsername:         MongoUsername,
		MongoPasswordLocation: MongoPasswordLocation,
	}

	learnerStdout, learnerStderr := StartLearner(learnerSession, learnerDistributedConfig)

	spectatorStdout, spectatorStderr := StartSpectator(spectatorSession, actorDistributedConfig, "spectator", 0.4)

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
			fmt.Printf("[learner] %s\n", msg)
		}
	}()

	actorClients := []*ssh.Client{}
	for host := range availableHosts {
		actorClients = append(actorClients, NewClient(host))
	}

	for i, client := range actorClients {
		defer client.Close()

		session, err := client.NewSession()
		if err != nil {
			panic("failed to create session: " + err.Error())
		}
		defer session.Close()

		stdout, stderr := StartActor(session, actorDistributedConfig, uuid.New().String(), epsilons[i])

		go func(client *ssh.Client) {
			for msg := range merge(stdout, stderr) {
				fmt.Printf("[actor %s] %s\n", client.RemoteAddr().String(), msg)
			}
		}(client)
	}

	updaterClient := NewClient(fmt.Sprintf("mimi.%s", FQDN))

	ticker := time.NewTicker(10 * time.Second)
	flag := make(chan bool)
	go func() {
		for {
			select {
			case <-ticker.C:
				copyTrainingGraphsToStaticSite(updaterClient)
			case <-flag:
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

	flag <- true
	close(flag)

	wg.Wait()
}
