#!/bin/bash

cd go
go build -o ../bin/find_servers cmd/find_servers/main.go 
go build -o ../bin/write_configs cmd/write_configs/main.go 
go build -o ../bin/hyperopt cmd/hyperopt/main.go 