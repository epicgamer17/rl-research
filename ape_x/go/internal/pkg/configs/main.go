package configs

type DistributedConfig struct {
	ReplayHost            string
	ReplayLearnerPort     int
	ReplayActorsPort      int
	MongoHost             string
	MongoPort             int
	MongoUsername         string
	MongoPasswordLocation string
	SpectatorHost         string
	ActorHosts            []string
	LearnerHost           string
	LearnerConfigFilename string
	ActorConfigFilename   string
	ReplayConfigFilename  string
	WithLearner           bool
	Alpha                 float64
	BaseNoisySigma        float64
}
