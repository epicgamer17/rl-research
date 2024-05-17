import time
import subprocess
from subprocess import Popen
from hyperopt import fmin, tpe, space_eval, hp, STATUS_OK
import pickle
import gc


def test(params):
    try:
        cmd = f"./bin/test_py"
        print("running cmd:", cmd)
        go_proc = Popen(cmd.split(" "), stdin=subprocess.PIPE, text=True)
        time.sleep(1)
        print("Training complete")
        return -params["score"]
    except Exception as e:
        print(e)
        return 0
    finally:
        stdout, stderr = go_proc.communicate("\n\n")
        print(f"stdout:{stdout}")
        print(f"stderr:{stderr}")


def objective(params):
    status = STATUS_OK
    loss = test(params)
    return {"loss": loss, "status": status}


def create_search_space():
    search_space = {"score": hp.choice("score", [10, 20, 30, 40, 50, 60, 70, 80, 90])}
    initial_best_config = []
    return search_space, initial_best_config


def main():
    ss, ibc = create_search_space()
    max_trials = 16
    trials_step = 1  # how many additional trials to do after loading the last ones

    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("./test.p", "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print(
            f"Rerunning from {len(trials.trials)} trials to {max_trials} (+{trials_step}) trials"
        )
    except:  # create a new trials object and start searching
        # trials = Trials()
        trials = None

    best = fmin(
        fn=objective,  # Objective Function to optimize
        space=ss,  # Hyperparameter's Search Space
        algo=tpe.suggest,  # Optimization algorithm (representative TPE)
        max_evals=max_trials,  # Number of optimization attempts
        trials=trials,  # Record the results
        # early_stop_fn=no_progress_loss(5, 1),
        trials_save_file="./classiccontrol_trials.p",
        points_to_evaluate=ibc,
        show_progressbar=False,
    )

    print(best)
    best_trial = space_eval(ss, best)
    print(best_trial)
    gc.collect()

    print("done")


if __name__ == "__main__":
    main()
