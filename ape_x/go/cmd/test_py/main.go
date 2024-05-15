package main

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"time"
)

func main() {
	ticker := time.NewTicker(time.Nanosecond)
	doneChannel := make(chan bool)

	go func() {
		for {
			select {
			case <-ticker.C:
				fmt.Println("[x] done")
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

	reader := bufio.NewReader(os.Stdin)
	_, err := reader.ReadString('\n')
	if err != nil {
		log.Println("error reading string", err)
	}
	fmt.Println("recieved stop signal, stopping")
	fmt.Println("main finished")
}
