package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"internal/pkg/configs"
	"internal/pkg/ssh_util"

	"gopkg.in/yaml.v3"
)

func CreateConfigsRemote(
	client *ssh_util.Client,
	config configs.DistributedConfig,
	learnerConfigFilename string,
	actorConfigFilename string,
	replayConfigFilename string,
	learnerConfigOutput string,
	actorConfigOutput string,
	replayConfigOutput string,
) {
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

	log.Println("Creating configs: ")

	// copy config files to remote, then run the config_generator.py script to inject the necessary distributed info
	commands := []string{
		"cd ~/rl-research/ape_x",
		"conda activate ml",
		fmt.Sprintf("echo '%s' > \"%s\"", string(learnerConfig), learnerConfigFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(actorConfig), actorConfigFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(replayConfig), replayConfigFilename),
		fmt.Sprintf("echo '%s' > \"%s\"", string(replayConfig), replayConfigOutput),
		fmt.Sprintf("python3 config_generator.py --replay_addr %s --replay_learner_port %d --replay_actors_port %d --storage_hostname %s --storage_port %d --storage_username %s --actor_base %s --learner_base %s --learner_output %s --actor_output %s", config.ReplayHost, config.ReplayLearnerPort, config.ReplayActorsPort, config.MongoHost, config.MongoPort, config.MongoUsername, actorConfigFilename, learnerConfigFilename, learnerConfigOutput, actorConfigOutput),
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

	actorsInitialSigmaFlag := flag.Float64("actors_initial_sigma", 0.9, "")
	actorsSigmaAlphaFlag := flag.Float64("actors_sigma_alpha", 20, "")

	learnerBaseFilenameFlag := flag.String("learner_config", "configs/learner_config_example.yaml", "")
	actorBaseFilenameFlag := flag.String("actor_config", "configs/actor_config_example.yaml", "")
	replayFilenameFlag := flag.String("replay_config", "configs/replay_config_example.yaml", "")
	hostsFilenameFlag := flag.String("hosts_file", "generated/hosts.yaml", "")

	learnerOutputFlag := flag.String("learner_output", "generated/learner_config.yaml", "")
	actorOutputFlag := flag.String("actor_output", "generated/actor_config.yaml", "")
	replayOutputFlag := flag.String("replay_output", "generated/replay_config.yaml", "")
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
		LearnerConfigFilename: *learnerOutputFlag,
		ActorConfigFilename:   *actorOutputFlag,
		ReplayConfigFilename:  *replayOutputFlag,
		Alpha:                 *actorsSigmaAlphaFlag,
		BaseNoisySigma:        *actorsInitialSigmaFlag,
	}

	client := ssh_util.NewClient(replayHost, *SSHUsernameFlag, "")
	CreateConfigsRemote(
		client,
		*distributedConfig,
		*learnerBaseFilenameFlag,
		*actorBaseFilenameFlag,
		*replayFilenameFlag,
		*learnerOutputFlag,
		*actorOutputFlag,
		*replayOutputFlag,
	)
	client.Close()

	out, err := yaml.Marshal(distributedConfig)
	if err != nil {
		panic(err)
	}

	os.WriteFile(*distributedOutputFlag, out, os.FileMode(0644))
}