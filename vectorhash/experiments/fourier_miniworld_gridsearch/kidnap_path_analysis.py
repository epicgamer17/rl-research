import os
import pickle
from common import generate_titles

import sys

sys.path.append("../..")
from agent_history import FourierVectorhashAgentHistory
from common import analyze_history_errors, write_animation

kidnap_results_dir = "kidnap_path_results"
kidnap_animations_dir = "kidnap_path_animations"
kidnap_plots_dir = "kidnap_path_plots"

os.makedirs(kidnap_animations_dir, exist_ok=True)
os.makedirs(kidnap_plots_dir, exist_ok=True)
titles = generate_titles()

for entry in os.listdir(kidnap_results_dir):
    with open(f"{kidnap_results_dir}/{entry}", "rb") as f:
        data: tuple[FourierVectorhashAgentHistory, list[int], list[int]] = pickle.load(
            f
        )
        history, pre_kidnap_path, post_kidnap_path = data
        write_animation(history, target_dir=kidnap_animations_dir, entry_name=entry)

        fig = analyze_history_errors(history, kidnap_t=240)

        i = int(entry.split(".")[0])
        fig.suptitle(titles[i])
        fig.savefig(f"{kidnap_plots_dir}/{i}.png")
