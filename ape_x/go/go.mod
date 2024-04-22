module apeX

go 1.21

require github.com/google/uuid v1.6.0

require (
	github.com/fatih/color v1.16.0 // indirect
	github.com/mattn/go-colorable v0.1.13 // indirect
	github.com/mattn/go-isatty v0.0.20 // indirect
	golang.org/x/crypto v0.22.0 // indirect
	golang.org/x/sys v0.19.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
)

require internal/pkg/ssh_util v1.0.0

replace internal/pkg/ssh_util => ./internal/pkg/ssh_util
