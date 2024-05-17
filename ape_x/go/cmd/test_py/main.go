package main

import (
	"fmt"
	"internal/pkg/ssh_util"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	defer log.Println("main finished normally")
	testClient := ssh_util.NewClient("mimi.cs.mcgill.ca", "ehuang", "test_client")
	cmdSess, err := testClient.Start("bash -c \"while sleep 0.1; do echo 'hello'; done\"", "pkill bash")
	if err != nil {
		panic(err)
	}
	defer testClient.Close()
	go cmdSess.StreamOutput("test_client")
	ticker := time.NewTicker(100 * time.Millisecond)
	doneChannel := make(chan bool)

	go func() {
		for {
			select {
			case <-ticker.C:
				fmt.Println("[x] test")
			case <-doneChannel:
				ticker.Stop()
				return
			}
		}
	}()
	defer func() {
		doneChannel <- true
		close(doneChannel)
	}()

	cancelChan := make(chan os.Signal, 1)
	// catch SIGETRM or SIGINTERRUPT
	signal.Notify(cancelChan, syscall.SIGTERM, syscall.SIGINT)

	sig := <-cancelChan
	log.Printf("Caught signal %v", sig)
	fmt.Println("stopping and cleaning up...")
}
