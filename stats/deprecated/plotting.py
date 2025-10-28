def default_plot_func(
    axs, key: str, values: list[dict], targets: dict, row: int, col: int
):
    axs[row][col].set_title(
        "{} | rolling average: {}".format(key, np.mean(values[-5:]))
    )
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, values)
    if key in targets and targets[key] is not None:
        axs[row][col].axhline(y=targets[key], color="r", linestyle="--")


def plot_scores(axs, key: str, values: list[dict], targets: dict, row: int, col: int):
    if len(values) == 0:
        return

    # print(values)
    scores = [value["score"] for value in values]
    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(scores[-100:])} | latest: {scores[-1]}"
    )

    # if scores are win/loss 1 or 0 only
    if not all(s in [-1, 0, 1] for s in scores):
        # If not, plot the raw scores
        x = np.arange(1, len(values) + 1)
        axs[row][col].plot(x, scores)
    else:
        print("scores are win/loss plotting a rolling average of the scores")
        scores = np.convolve(scores, np.ones(100) / 100, mode="valid")
        x = np.arange(1, len(scores) + 1)
        axs[row][col].plot(x, scores)

    has_max_scores = "max_score" in values[0]
    has_min_scores = "min_score" in values[0]
    assert (
        has_max_scores == has_min_scores
    ), "Both max_scores and min_scores must be provided or not provided"

    if has_max_scores:
        max_scores = [value["max_score"] for value in values]
        min_scores = [value["min_score"] for value in values]
        axs[row][col].fill_between(x, min_scores, max_scores, alpha=0.5)

    has_target_model_updates = "target_model_updated" in values[0]
    has_model_updates = "model_updated" in values[0]

    if has_target_model_updates:
        weight_updates = [value["target_model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="black",
                    linestyle="dotted",
                    # label="Target Model Weight Update",
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    # label="Model Weight Update",
                )

    axs[row][col].set_xlabel("Game")
    axs[row][col].set_ylabel("Score")

    axs[row][col].set_xlim(1, len(scores))

    if len(scores) > 1:
        best_fit_x, best_fit_y = np.polyfit(x, scores, 1)
        axs[row][col].plot(
            x,
            best_fit_x * x + best_fit_y,
            color="g",
            label="Best Fit Line",
            linestyle="dotted",
        )

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Score: {}".format(targets[key]),
        )

    axs[row][col].legend()


def plot_loss(axs, key: str, values: list[dict], targets: dict, row: int, col: int):
    loss = [value["loss"] for value in values]
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, loss)

    has_target_model_updates = "target_model_updated" in values[0]
    has_model_updates = "model_updated" in values[0]

    if has_target_model_updates:
        weight_updates = [value["target_model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="black",
                    linestyle="dotted",
                    # label="Target Model Weight Update",
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    # label="Model Weight Update",
                )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(loss[-5:])} | latest: {loss[-1]}"
    )

    axs[row][col].set_xlabel("Time Step")
    axs[row][col].set_ylabel("Loss")

    axs[row][col].set_xlim(1, len(values))

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Score: {}".format(targets[key]),
        )

    axs[row][col].legend()


def plot_exploitability(
    axs, key: str, values: list[dict], targets: dict, row: int, col: int
):
    if len(values) == 0:
        return
    exploitability = [abs(value["exploitability"]) for value in values]
    print(values)
    rolling_averages = [
        np.mean(exploitability[max(0, i - 5) : i])
        for i in range(1, len(exploitability) + 1)
    ]
    # print(rolling_averages)
    x = np.arange(1, len(values) + 1)
    axs[row][col].plot(x, rolling_averages)
    axs[row][col].plot(x, exploitability)

    has_target_model_updates = "target_model_updated" in values[0]
    has_model_updates = "model_updated" in values[0]

    if has_target_model_updates:
        weight_updates = [value["target_model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="black",
                    linestyle="dotted",
                    # label="Target Model Weight Update",
                )

    if has_model_updates:
        weight_updates = [value["model_updated"] for value in values]
        for i, weight_update in enumerate(weight_updates):
            if weight_update:
                axs[row][col].axvline(
                    x=i,
                    color="gray",
                    linestyle="dotted",
                    # label="Model Weight Update",
                )

    if len(rolling_averages) > 1:
        best_fit_x, best_fit_y = np.polyfit(x, rolling_averages, 1)
        axs[row][col].plot(
            x,
            best_fit_x * x + best_fit_y,
            color="g",
            label="Best Fit Line",
            linestyle="dotted",
        )

    axs[row][col].set_title(
        f"{key} | rolling average: {np.mean(exploitability[-5:])} | latest: {exploitability[-1]}"
    )

    axs[row][col].set_xlabel("Game")
    axs[row][col].set_ylabel("Exploitability (rolling average)")

    axs[row][col].set_xscale("log")
    axs[row][col].set_yscale("log")

    axs[row][col].set_xlim(1, len(values))
    # axs[row][col].set_ylim(0.01, 10)
    # axs[row][col].set_ylim(
    #     -(10 ** math.ceil(math.log10(abs(min_exploitability)))),
    #     10 ** math.ceil(math.log10(max_exploitability)),
    # )

    # axs[row][col].set_yticks(
    #     [
    #         -(10**i)
    #         for i in range(
    #             math.ceil(math.log10(abs(min_exploitability))),
    #             math.floor(math.log10(abs(min_exploitability))) - 1,
    #             -1,
    #         )
    #         if -(10**i) < min_exploitability
    #     ]
    #     + [0]
    #     + [
    #         10**i
    #         for i in range(
    #             math.ceil(math.log10(max_exploitability)),
    #             math.floor(math.log10(max_exploitability)) + 1,
    #         )
    #         if 10**i > max_exploitability
    #     ]
    # )

    if key in targets and targets[key] is not None:
        axs[row][col].axhline(
            y=targets[key],
            color="r",
            linestyle="dashed",
            label="Target Exploitability: {}".format(targets[key]),
        )

    axs[row][col].legend()


def plot_trials(scores: list, file_name: str, final_trial: int = 0):
    fig, axs = plt.subplots(
        1,
        1,
        figsize=(10, 5),
        squeeze=False,
    )
    if final_trial > 0:
        x = np.arange(1, final_trial + 1)
        scores = scores[:final_trial]
    else:
        x = np.arange(1, len(scores) + 1)
    axs[0][0].scatter(x, scores)
    best_fit_x, best_fit_y = np.polyfit(x, scores, 1)
    axs[0][0].plot(
        x,
        best_fit_x * x + best_fit_y,
        color="g",
        label="Best Fit Line",
        linestyle="dotted",
    )

    fig.suptitle("Score of Hyperopt trials over time for Rainbow DQN on CartPole-v1")
    axs[0][0].set_xlabel("Trial")
    axs[0][0].set_ylabel("Score")
    plt.savefig(f"./graphs/{file_name}.png")
    plt.show()
    plt.close(fig)


stat_keys_to_plot_funcs = {
    "test_score": plot_scores,
    "test_score_vs_random": plot_scores,
    "score": plot_scores,
    "policy_loss": plot_loss,
    "value_loss": plot_loss,
    "reward_loss": plot_loss,
    "l2_loss": plot_loss,
    "loss": plot_loss,
    "rl_loss": plot_loss,
    "sl_loss": plot_loss,
    "exploitability": plot_exploitability,  # should this be plot_scores?
}


def plot_graphs(
    stats: dict,
    targets: dict,
    step: int,
    frames_seen: int,
    time_taken: float,
    model_name: str,
    dir: str = "./checkpoints/graphs",
):
    num_plots = len(stats)
    sqrt_num_plots = math.ceil(np.sqrt(num_plots))
    fig, axs = plt.subplots(
        sqrt_num_plots,
        sqrt_num_plots,
        figsize=(10 * sqrt_num_plots, 5 * sqrt_num_plots),
        squeeze=False,
    )

    hours = int(time_taken // 3600)
    minutes = int((time_taken % 3600) // 60)
    seconds = int(time_taken % 60)

    fig.suptitle(
        "training stats | training step {} | frames seen {} | time taken {} hours {} minutes {} seconds".format(
            step, frames_seen, hours, minutes, seconds
        )
    )

    for i, (key, values) in enumerate(stats.items()):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots
        if len(values) == 0:
            print(f"Skipping {key}...")
            continue
        print(f"Plotting {key}...")
        if key in stat_keys_to_plot_funcs:
            stat_keys_to_plot_funcs[key](axs, key, values, targets, row, col)
        else:
            default_plot_func(axs, key, values, targets, row, col)

    for i in range(num_plots, sqrt_num_plots**2):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots
        fig.delaxes(axs[row][col])

    # plt.show()
    assert os.path.exists(dir), f"Directory {dir} does not exist"
    plt.savefig("{}/{}.png".format(dir, model_name))

    plt.close(fig)


def plot_comparisons(
    stats: list[dict],
    model_name: str,
    dir: str = "./checkpoints/graphs",
):
    num_plots = len(stats[0])
    sqrt_num_plots = math.ceil(np.sqrt(num_plots))
    fig, axs = plt.subplots(
        sqrt_num_plots,
        sqrt_num_plots,
        figsize=(10 * sqrt_num_plots, 5 * sqrt_num_plots),
        squeeze=False,
    )

    fig.suptitle("Comparison of training stats")

    for i, (key, _) in enumerate(stats[0].items()):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots
        # max_value = float("-inf")
        # min_value = float("inf")
        max_len = 0
        for s in stats:
            values = s[key]
            # print(values)
            max_len = max(max_len, len(values))
            print(max_len)
            # max_value = max(max_value, max(values))
            # min_value = min(min_value, min(values))
            if key in stat_keys_to_plot_funcs:
                stat_keys_to_plot_funcs[key](axs, key, values, {}, row, col)
                axs[row][col].set_xlim(0, max_len)
            else:
                default_plot_func(axs, key, values, {}, row, col)

        # axs[row][col].set_ylim(min_value, max_value)

    for i in range(num_plots, sqrt_num_plots**2):
        row = i // sqrt_num_plots
        col = i % sqrt_num_plots
        fig.delaxes(axs[row][col])

    # plt.show()
    os.makedirs(dir, exist_ok=True)
    plt.savefig("{}/{}.png".format(dir, model_name))

    plt.close(fig)
