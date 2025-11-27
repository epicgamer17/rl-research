from hyperopt import space_eval
import dill as pickle
import os
import numpy as np

from stats.deprecated.plotting import plot_trials


def hyperopt_analysis(
    data_dir: str,
    file_name: str,
    viable_trial_threshold: int,
    step: int,
    final_trial: int = 0,
    eval_method: str = "final_score",
):
    trials = pickle.load(open(f"{data_dir}/{file_name}.p", "rb"))
    if final_trial > 0:
        print("Number of trials: {}".format(final_trial))
    else:
        print("Number of trials: {}".format(len(trials.trials)))
    # losses.sort()
    # print(len(os.listdir(f"{data_dir}/checkpoints")) - 1)
    # print(len(trials.trials))

    checkpoints = os.listdir(f"{data_dir}/checkpoints")
    checkpoints.remove("videos") if "videos" in checkpoints else None
    checkpoints.remove(".DS_Store") if ".DS_Store" in checkpoints else None
    checkpoints.sort(key=lambda x: int(x.split("_")[-1]))
    if final_trial > 0:
        checkpoints = checkpoints[:final_trial]

    viable_throughout_trials = []
    final_rolling_averages = []
    final_std_devs = []
    scores = []
    losses = []
    failed_trials = 0
    for i, trial in enumerate(trials.trials):
        losses.append(trial["result"]["loss"])
        if final_trial > 0 and i >= final_trial:
            break
        # print(trial["result"]["status"])
        if trial["result"]["status"] == "fail":
            failed_trials += 1
            final_rolling_averages.append(trial["result"]["loss"])
            scores.append(trial["result"]["loss"])
            final_std_devs.append(trial["result"]["loss"])
        else:
            # print(checkpoints[i - failed_trials])
            # print(failed_trials)
            # if os.path.exists(
            #     f"{data_dir}/checkpoints/{checkpoints[i - failed_trials]}/step_{step}/graphs_stats/stats.pkl"
            # ):
            stats = pickle.load(
                open(
                    f"{data_dir}/checkpoints/{checkpoints[i - failed_trials]}/step_{step}/graphs_stats/stats.pkl",
                    "rb",
                )
            )
            max_score = 0

            # print([stat_dict["score"] for stat_dict in stats["test_score"][-5:]])
            final_rolling_averages.append(
                np.around(
                    np.mean(
                        [stat_dict["score"] for stat_dict in stats["test_score"][-5:]]
                    ),
                    1,
                )
            )

            final_std_devs.append(
                np.around(
                    np.std(
                        [stat_dict["score"] for stat_dict in stats["test_score"][-5:]]
                    ),
                    1,
                )
            )

            for stat_dict in stats["test_score"]:
                if stat_dict["max_score"] > max_score:
                    max_score = stat_dict["max_score"]

            if max_score > viable_trial_threshold:
                viable_throughout_trials.append(max_score)

            if eval_method == "final_score":
                score = -trial["result"]["loss"]
            elif (
                eval_method == "rolling_average"
                or eval_method == "final_score_rolling_average"
            ):
                score = stats["test_score"][-1]["score"]
            scores.append(score)

    plot_trials(
        scores,
        file_name,
        final_trial=final_trial,
    )

    res = [
        list(x)
        for x in zip(
            *sorted(
                zip(losses, scores, final_rolling_averages, final_std_devs),
                key=itemgetter(0),
            )
        )
    ]
    losses = res[0]
    scores = res[1]
    final_rolling_averages = res[2]
    final_std_devs = res[3]
    viable_trials = [score for score in scores if score > viable_trial_threshold]

    print("Failed trials: ~{}%".format(round(failed_trials / len(scores) * 100)))

    print(
        "Viable trials (based on final score): ~{}%".format(
            round(len(viable_trials) / len(scores) * 100)
        )
    )
    print(
        "Viable trials (throughout training): ~{}%".format(
            round(len(viable_throughout_trials) / len(scores) * 100)
        )
    )

    print("Losses: {}".format(losses))
    print("Scores: {}".format(scores))
    print("Final rolling averages: {}".format(final_rolling_averages))
    print("Final standard deviations: {}".format(final_std_devs))

    print("Max loss: {}".format(max(losses)))
    print("Max score: {}".format(max(scores)))
    print("Max final rolling average: {}".format(max(final_rolling_averages)))
    print("Max final standard deviation: {}".format(max(final_std_devs)))

    print("Average loss: {}".format(np.mean(losses)))
    print("Average score: {}".format(np.mean(scores)))
    print("Average final rolling average: {}".format(np.mean(final_rolling_averages)))
    print("Average final standard deviation: {}".format(np.mean(final_std_devs)))

    viable_final_rolling_averages = [
        final_rolling_averages[i]
        for i, loss in enumerate(scores)
        if loss > viable_trial_threshold
    ]

    viable_std_devs = [
        final_std_devs[i]
        for i, loss in enumerate(scores)
        if loss > viable_trial_threshold
    ]

    print(
        "Average score of viable trials (based on final score): {}".format(
            np.mean(viable_trials)
        )
    )
    print(
        "Average final rolling average of viable trials (based on final score): {}".format(
            np.mean(viable_final_rolling_averages)
        )
    )
    print(
        "Average final standard deviation of viable trials (based on final score): {}".format(
            np.mean(viable_std_devs)
        )
    )


def graph_hyperparameter_importance(
    data_dir: str, trials_file: str, search_space_file: str, viable_trial_threshold: int
):
    with open(f"{data_dir}/{trials_file}", "rb") as f:
        trials = pickle.load(f)
    print(trials)

    search_space = pickle.load(open(f"./search_spaces/{search_space_file}", "rb"))

    values_dict = defaultdict(list)
    scores = []
    for trial in trials.trials:
        for key, value in space_eval(trial["misc"]["vals"], search_space).items():
            values_dict[key].append(value[0])
        scores.append(-trial["result"]["loss"])

    df = pd.DataFrame(values_dict)
    x_cols = df.columns
    df["scores"] = scores
    # print(df)
    df = df[df["scores"] > viable_trial_threshold]

    for col in x_cols:
        if col == "loss_function":
            continue
        plt = df.plot(x=col, y="scores", kind="scatter")
        grouped = df.groupby(col)["scores"]
        medians = grouped.median()
        means = grouped.mean()
        stddev = grouped.std()

        if not (col == "kernel_initializer" or col == "activation"):
            # plt.fill_between(medians.index, medians.values-stddev, medians.values+stddev, color="#00F0F0")
            plt.plot(means.index, means.values, color="#00FFFF")
        else:
            plt.scatter(means.index, means.values, c="#00FFFF")
        # plt.add_line
