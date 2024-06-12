package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/fatih/color"
	"gopkg.in/yaml.v3"
)

const SSH_ARGS = "-oStrictHostKeyChecking=no -oConnectTimeout=5"
const USERNAME = "ehuang"
const FQDN = "cs.mcgill.ca"

func GenerateHosts(username string, indicesToExclude map[int]bool) <-chan string {
	c := make(chan string)
	go func() {
		for i := 1; i <= 33; i++ {
			if _, ok := indicesToExclude[i]; ok {
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
		args := append(strings.Split(SSH_ARGS, " "), connectionString, "who | cut -d \" \" -f1 | sort -n | uniq -c | wc -l")
		cmd := exec.Command("ssh", args...)

		go func(h string) {
			defer wg.Done()
			output, err := cmd.Output()
			if err != nil {
				color.Red("  %s is not online", h)
				return
			}
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
		}(host)
	}

	go func() {
		wg.Wait()
		done <- true
		close(c)
		close(done)
	}()

	return c, done
}

type arrayFlagInt []int

func (arr *arrayFlagInt) String() string {
	ret := ""
	for i := range *arr {
		ret += fmt.Sprintf("%d ", i)
	}
	return ret
}

func (arr *arrayFlagInt) Set(value string) error {
	v, err := strconv.Atoi(value)
	if err != nil {
		return err
	}

	*arr = append(*arr, v)
	return nil
}

func main() {
	var excludeFlag arrayFlagInt
	flag.Var(&excludeFlag, "exclude", "")
	outputFilenameFlag := flag.String("output", "../generated/hosts.yaml", "")
	usernameFlag := flag.String("ssh_username", USERNAME, "")
	machinesFlag := flag.Int("machines", 4, "")

	flag.Parse()

	indicesToExclude := make(map[int]bool)

	for _, i := range []int{13, 16, 17} {
		// 13 - /home/2023/ehuang/.conda/envs/ml/lib/python3.11/site-packages/torch/cuda/__init__.py:118: UserWarning: CUDA initialization: CUDA unknown error - this may be due to an incorrectly set up environment, e.g. changing env variable CUDA_VISIBLE_DEVICES after program start. Setting the available devices to be zero. (Triggered internally at ../c10/cuda/CUDAFunctions.cpp:108.) \n return torch._C._cuda_getDeviceCount() > 0
		// 16, 17 - bad driver version
		indicesToExclude[i] = true
	}

	for _, i := range excludeFlag {
		indicesToExclude[i] = true
	}

	fmt.Println("excluding:", excludeFlag)

	hostsCh, done := FindFreeServers(GenerateHosts(*usernameFlag, indicesToExclude))
	hosts := []string{}

	go func() {
		for host := range hostsCh {
			hosts = append(hosts, host)
		}
	}()

	<-done

	if len(hosts) < *machinesFlag {
		fmt.Println("not enough actors to run.")
		os.Exit(1)
	}
	truncated := hosts[:*machinesFlag]
	enc, err := yaml.Marshal(truncated)
	if err != nil {
		panic(err)
	}

	os.WriteFile(*outputFilenameFlag, enc, os.FileMode(0644))
}
