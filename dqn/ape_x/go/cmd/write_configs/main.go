package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"strings"

	"internal/pkg/configs"
	"internal/pkg/ssh_util"

	"gopkg.in/yaml.v3"
)

func getHostname() string {
	cmd := exec.Command("/bin/hostname", "-f")
	var buf bytes.Buffer
	cmd.Stdout = &buf
	err := cmd.Run()
	if err != nil {
		log.Fatal(err)
	}
	fqdn := buf.String()
	fqdn = fqdn[:len(fqdn)-1]
	return fqdn
}

func CreateConfigsRemote(
	client *ssh_util.Client,
	config configs.DistributedConfig,
	learnerConfigBaseFilename string,
	actorConfigBaseFilename string,
	learnerConfigOutput string,
	actorConfigOutput string,
) {
	learnerConfig, err := os.ReadFile(learnerConfigBaseFilename)
	if err != nil {
		panic(err)
	}
	actorConfig, err := os.ReadFile(actorConfigBaseFilename)
	if err != nil {
		panic(err)
	}

	log.Println("Creating configs: ")

	// copy base files to remote, then run the config_generator.py script (remotely)
	// to inject the necessary distributed info

	worldSize := len(config.ActorHosts) + 3
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("echo '%s' > \"%s\"", string(learnerConfig), learnerConfigBaseFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(actorConfig), actorConfigBaseFilename),
		fmt.Sprintf("python3 config_generator.py --world_size %d --master_addr %s --replay_addr %s --storage_addr %s --rpc_port %d --pg_port %d", worldSize, config.MasterHost, config.ReplayHost, config.StorageHost, config.RPCPort, config.PGPort),
	}

	cmd := strings.Join(commands, "; ")
	out, err := client.Run(cmd)
	if err != nil {
		log.Println("err: ", err)
		log.Println("out: ", string(out))
		panic(err)
	}
}

const USERNAME = "ehuang"
const ReplayLearnerPort = 5554
const ReplayActorPort = 5555
const MongoPort = 5553
const MongoUsername = "ezra"
const MongoPasswordLocation = "~/mongodb/mongodb_admin_password"

func main() {
	SSHUsernameFlag := flag.String("ssh_username", USERNAME, "")

	learnerBaseFilenameFlag := flag.String("learner_config", "configs/learner_config_example.yaml", "")
	actorBaseFilenameFlag := flag.String("actor_config", "configs/actor_config_example.yaml", "")
	hostsFilenameFlag := flag.String("hosts_file", "generated/hosts.yaml", "")

	learnerOutputFlag := flag.String("learner_output", "generated/learner_config.yaml", "")
	actorOutputFlag := flag.String("actor_output", "generated/actor_config.yaml", "")
	distributedOutputFlag := flag.String("distributed_output", "generated/distributed_config.yaml", "")

	withLearnerFlag := flag.Bool("with_learner", false, "")
	flag.Parse()

	hostsFileContent, err := os.ReadFile(*hostsFilenameFlag)
	if err != nil {
		panic(err)
	}
	hosts := []string{}
	yaml.Unmarshal(hostsFileContent, &hosts)
	log.Println("Using hosts: ", hosts)

	replayHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]
	storageHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]

	learnerHost := ""
	masterHost := ""
	if *withLearnerFlag {
		learnerHost, hosts = hosts[len(hosts)-1], hosts[:len(hosts)-1]
		masterHost = learnerHost
	} else {
		masterHost = getHostname()
	}

	distributedConfig := configs.DistributedConfig{
		LearnerHost:           learnerHost,
		ReplayHost:            replayHost,
		StorageHost:           storageHost,
		MasterHost:            masterHost,
		RPCPort:               3333, // hardcoded for now
		PGPort:                3334, // hardcoded for now
		ActorHosts:            hosts,
		LearnerConfigFilename: *learnerOutputFlag,
		ActorConfigFilename:   *actorOutputFlag,
		WithLearner:           *withLearnerFlag,
	}

	client := ssh_util.NewClient(replayHost, *SSHUsernameFlag, "")
	CreateConfigsRemote(
		client,
		distributedConfig,
		*learnerBaseFilenameFlag,
		*actorBaseFilenameFlag,
		*learnerOutputFlag,
		*actorOutputFlag,
	)
	client.Close()

	out, err := yaml.Marshal(distributedConfig)
	if err != nil {
		panic(err)
	}

	// write a local copy for debugging
	os.WriteFile(*distributedOutputFlag, out, os.FileMode(0644))
}
