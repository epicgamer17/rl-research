package configs

type DistributedConfig struct {
	LearnerHost           string
	ReplayHost            string
	StorageHost           string
	MasterHost            string
	RPCPort               int
	PGPort                int
	ActorHosts            []string
	LearnerConfigFilename string
	ActorConfigFilename   string
	WithLearner           bool
}
