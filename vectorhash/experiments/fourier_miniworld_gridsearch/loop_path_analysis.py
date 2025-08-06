import os
import pickle
from common import generate_titles

import sys

sys.path.append("../..")
from agent_history import FourierVectorhashAgentHistory
from common import analyze_history_errors, write_animation


loop_results_dir = "loop_path_results_aug_6_fast"
loop_animations_dir = "loop_path_animations_aug_6_fast"
loop_plots_dir = "loop_path_plots_aug_6_fast"

os.makedirs(loop_animations_dir, exist_ok=True)
os.makedirs(loop_plots_dir, exist_ok=True)
titles = generate_titles()

for entry in os.listdir(loop_results_dir):
    with open(f"{loop_results_dir}/{entry}", "rb") as f:
        data: tuple[FourierVectorhashAgentHistory, list[int]] = pickle.load(f)
        history, path = data
        write_animation(history, loop_animations_dir, entry)
        fig = analyze_history_errors(history)

        i = int(entry.split(".")[0])
        fig.suptitle(titles[i])
        fig.savefig(f"{loop_plots_dir}/{i}.png")
