import os
import pickle

import sys

sys.path.append("../..")
from agent_history import FourierVectorhashAgentHistory

results_dir = "loop_path_results"
animations_dir = "loop_path_animations"

os.makedirs(animations_dir, exist_ok=True)

for entry in os.listdir(results_dir):
    with open(f"{results_dir}/{entry}", "rb") as f:
        data: tuple[FourierVectorhashAgentHistory, list[int]] = pickle.load(f)
        history, path = data
        anim = history.make_image_video()
        anim.save(
            f"{animations_dir}/{entry.split('.')[0]}.gif",
            progress_callback=lambda step, total: print(f"frame {step+1}/{total}"),
        )
