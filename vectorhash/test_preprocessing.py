# test_preprocessor.py

import numpy as np
import torch

# 1) Import your PreprocessingCNN class
from preprocessing_cnn import PreprocessingCNN
from miniworld.params import DEFAULT_PARAMS

import gymnasium as gym




def main():
    # 2) Make a dummy “observation”:
    #    Suppose some env would give you an uint8 image of shape (60, 80, 3).
    #    We’ll fill it with random values [0..255].
    dummy_observation = np.random.randint(
        low=0, high=256, size=(60, 80, 3), dtype=np.uint8
    )
    
    # 3) Instantiate PreprocessingCNN on CPU (target size 84×84, latent_dim=128)
    device = torch.device("cpu")
    preproc = PreprocessingCNN(
        device=device,
        latent_dim=128,
        input_channels=3,
        target_size=(84, 84),
        model_path=None,   # None → uses random weights (no pretrained encoder)
    )

    # 4) Call .encode(...) on the dummy observation
    latent_vector = preproc.encode(dummy_observation)

    # 5) Print out the result’s type, shape, and a small slice of values
    print(f"latent_vector type: {type(latent_vector)}")
    print(f"latent_vector device: {latent_vector.device}")
    print(f"latent_vector shape: {tuple(latent_vector.shape)}")
    print(f"first 10 elements: {latent_vector[:10].tolist()}")

    # 6) Check that the latent values are floats
    print(f"  dtype: {latent_vector.dtype}")
    
    params = DEFAULT_PARAMS.copy().no_random()
    env = gym.make("MiniWorld-Maze-v0", max_episode_steps=-1, domain_rand=False, params=params)
    obs, info = env.reset()
    # obs will be a (60,80,3) uint8 array.

    preproc = PreprocessingCNN(device=device, latent_dim=128, input_channels=3, target_size=(84,84))
    latent_real = preproc.encode(obs)
    print("Real env latent shape:", latent_real.shape)
    
    
    

if __name__ == "__main__":
    main()
