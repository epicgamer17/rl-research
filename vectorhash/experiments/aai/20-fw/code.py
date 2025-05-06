import random
from animalai.environment import AnimalAIEnvironment
from test_utils import generate_animalai_path
from wrappers import CustomUnityToGymWrapper
from animalai_agent import AnimalAIVectorhashAgent
from vectorhash import build_vectorhash_architecture
from smoothing import PolynomialSmoothing
import random
from shifts import RatShiftWithCompetitiveAttractorDynamics

### vhash
shapes = [(5, 5, 5), (8, 8, 8)]
model = build_vectorhash_architecture(
    shapes,
    N_h=1200,
    input_size=84 * 84,
    initalization_method="by_sparsity",
    smoothing=PolynomialSmoothing(k=1.5),
    shift=RatShiftWithCompetitiveAttractorDynamics(
        sigma_xy=0.3, sigma_theta=0.3, inhibition_constant=0.004, delta_gamma=1
    ),
    limits=(40, 40, 360),
    relu=True,
    percent_nonzero_relu=0.2,
)


### animalai
aai_seed = 0
port = 5005 + random.randint(
    0, 1000
)  # uses a random port to avoid problems if a previous version exits slowly
env_path = "/home/ezrahuang/AAI/LINUX/AAI.x86_64"
configuration_file = "./animal_ai_environments/yroom.yaml"
watch = True

aai_env = AnimalAIEnvironment(
    file_name=env_path,  # Path to the environment
    seed=aai_seed,  # seed for the pseudo random generators
    arenas_configurations=configuration_file,
    play=False,  # note that this is set to False for training
    base_port=port,  # the port to use for communication between python and the Unity environment
    inference=False,  # set to True if you want to watch the agent play
    useCamera=True,  # set to False if you don't want to use the camera (no visual observations)
    resolution=84,
    useRayCasts=False,  # set to True if you want to use raycasts
    no_graphics=False,  # set to True if you don't want to use the graphics ('headless' mode)
    timescale=1,
)

env = CustomUnityToGymWrapper(
    aai_env, uint8_visual=False, allow_multiple_obs=True, flatten_branched=True
)  # the wrapper for the environment
agent = AnimalAIVectorhashAgent(model, env)
# path = generate_animalai_path(path_length=20)

path = [3] * 20
agent.test_path(path)