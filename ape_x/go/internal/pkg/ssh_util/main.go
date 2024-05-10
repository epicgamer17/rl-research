package ssh_util

import (
	"bufio"
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"
)

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

func Reader(reader io.Reader, finished *bool) <-chan string {
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
					*finished = true
					break
				}
			}
		}
	}()
	return ch
}

type CommandSession struct {
	cmd            string
	killCommand    string
	stdout         <-chan string
	stderr         <-chan string
	stdoutFinished bool
	stderrFinished bool
}

type Client struct {
	Name      string
	SSHClient *ssh.Client
	sessions  map[*ssh.Session]*CommandSession
}

func NewClient(host string, username string, name string) *Client {
	config := NewSSHConfig(username)
	address := net.JoinHostPort(host, "22")
	client, err := ssh.Dial("tcp", address, config)
	if err != nil {
		panic(fmt.Sprintf("failed to dial %s@%s: %s", host, username, err.Error()))
	}

	return &Client{
		Name:      name,
		SSHClient: client,
		sessions:  make(map[*ssh.Session]*CommandSession),
	}
}

func (c *Client) Run(cmd string) ([]byte, error) {
	session, err := c.SSHClient.NewSession()
	if err != nil {
		fmt.Println("failed to create session: " + err.Error())
		return []byte{}, err
	}
	defer session.Close()

	fmt.Printf("[%s] Running command: %s\n", c.Name, cmd)
	return session.CombinedOutput(cmd)
}

func (c *Client) Start(cmd string, killCmd string) (*CommandSession, error) {
	session, err := c.SSHClient.NewSession()
	if err != nil {
		fmt.Println("failed to create session: " + err.Error())
		return nil, err
	}

	fmt.Printf("[%s] Starting command: %s\n", c.Name, cmd)
	commandSession := c.NewCommandSession(session, cmd, killCmd)
	if err = session.Start(cmd); err != nil {
		fmt.Printf("Error running command %s on %s: %s\n", cmd, c.Name, err)
		return nil, err
	}

	c.sessions[session] = commandSession
	return commandSession, nil
}

func (c *Client) Close() {
	for k, v := range c.sessions {
		if !v.stderrFinished || !v.stdoutFinished {
			fmt.Printf("cleaning up unfinished session on %s\n", c.SSHClient.Conn.RemoteAddr())
			out, err := c.Run(v.killCommand)
			fmt.Printf("output %s: %s\n", c.Name, out)
			fmt.Printf("err %s: %s\n", c.Name, err)
		}
		if !v.stderrFinished || !v.stdoutFinished {
			fmt.Printf("warning: command %s on host %s never terminated\n", v.cmd, c.SSHClient.Conn.RemoteAddr())
		}
		if err := k.Close(); err != nil {
			fmt.Println("Error closing session: ", err)
		}
	}
	c.SSHClient.Close()
}

func (c *Client) NewCommandSession(session *ssh.Session, cmd string, killCmd string) *CommandSession {
	cmdSess := &CommandSession{
		cmd:            cmd,
		killCommand:    killCmd,
		stdoutFinished: false,
		stderrFinished: false,
	}
	stderrPipe, err := session.StderrPipe()
	if err != nil {
		panic("failed to create stderr pipe: " + err.Error())
	}
	stdoutPipe, err := session.StdoutPipe()
	if err != nil {
		panic("failed to create stdout pipe: " + err.Error())
	}

	cmdSess.stdout = Reader(stdoutPipe, &cmdSess.stdoutFinished)
	cmdSess.stderr = Reader(stderrPipe, &cmdSess.stderrFinished)

	return cmdSess
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

func (cs *CommandSession) StreamOutput(prefix string) {
	go func() {
		for msg := range merge(cs.stdout, cs.stderr) {
			fmt.Printf("%s %s\n", prefix, msg)
		}
	}()
}
