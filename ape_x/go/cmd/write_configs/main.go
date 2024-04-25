package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"internal/pkg/configs"
	"internal/pkg/ssh_util"

	"gopkg.in/yaml.v3"
)

var OutputDir = "generated"

func CreateConfigsRemote(client *ssh_util.Client, config configs.DistributedConfig, learnerConfigFilename string, actorConfigFilename string, replayConfigFilename string) {
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

	// write config files to remote, then run the config_generator.py script to inject the necessary info
	commands := []string{
		"cd ~/rl-research/ape_x/go",
		"conda activate ml",
		fmt.Sprintf("echo '%s' > \"%s\"", string(learnerConfig), learnerConfigFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(actorConfig), actorConfigFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(replayConfig), replayConfigFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(replayConfig), "../generated/replay_config.yaml"),
		fmt.Sprintf("python3 ../config_generator.py --replay_addr %s --replay_learner_port %d --replay_actors_port %d --storage_hostname %s --storage_port %d --storage_username %s --actor_base %s --learner_base %s --output %s", config.ReplayHost, config.ReplayLearnerPort, config.ReplayActorsPort, config.MongoHost, config.MongoPort, config.MongoUsername, actorConfigFilename, learnerConfigFilename, fmt.Sprintf("../%s", OutputDir)),
	}

	cmd := strings.Join(commands, "; ")
	out, err := client.Run(cmd)
	if err != nil {
		fmt.Println("err: ", err)
		fmt.Println("out: ", string(out))
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
	learnerBaseFilenameFlag := flag.String("learner_config", "../configs/learner_config_example.yaml", "")
	actorBaseFilenameFlag := flag.String("actor_config", "../configs/actor_config_example.yaml", "")
	replayFilenameFlag := flag.String("replay_config", "../configs/replay_config_example.yaml", "")
	hostsFilenameFlag := flag.String("hosts_file", "../generated/hosts.yaml", "")
	outFilenameFlag := flag.String("output", "../generated/distributed_config.yaml", "")
	withLearnerFlag := flag.Bool("with_learner", false, "")
	hostsFileContent, err := os.ReadFile(*hostsFilenameFlag)
	if err != nil {
		panic(err)
	}
	hosts := []string{}

	yaml.Unmarshal(hostsFileContent, &hosts)

	fmt.Println("Using hosts: ", hosts)

	replayHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]
	mongoHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]
	spectatorHost, hosts := hosts[len(hosts)-1], hosts[:len(hosts)-1]

	learnerHost := ""
	if *withLearnerFlag {
		learnerHost, hosts = hosts[len(hosts)-1], hosts[:len(hosts)-1]
	}

	distributedConfig := &configs.DistributedConfig{
		ReplayHost:            replayHost,
		ReplayLearnerPort:     ReplayLearnerPort,
		ReplayActorsPort:      ReplayActorPort,
		MongoHost:             mongoHost,
		MongoPort:             MongoPort,
		MongoUsername:         MongoUsername,
		MongoPasswordLocation: MongoPasswordLocation,
		LearnerHost:           learnerHost,
		WithLearner:           *withLearnerFlag,
		ActorHosts:            hosts,
		SpectatorHost:         spectatorHost,
		LearnerConfigFilename: fmt.Sprintf("%s/learner_config.yaml", OutputDir),
		ActorConfigFilename:   fmt.Sprintf("%s/actor_config.yaml", OutputDir),
		ReplayConfigFilename:  fmt.Sprintf("%s/replay_config.yaml", OutputDir),
	}

	client := ssh_util.NewClient(replayHost, USERNAME, "")
	CreateConfigsRemote(client, *distributedConfig, *learnerBaseFilenameFlag, *actorBaseFilenameFlag, *replayFilenameFlag)
	client.Close()

	out, err := yaml.Marshal(distributedConfig)
	if err != nil {
		panic(err)
	}
	os.WriteFile(*outFilenameFlag, out, os.FileMode(0644))
}
