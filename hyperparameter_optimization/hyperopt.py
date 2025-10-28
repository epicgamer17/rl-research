import dill as pickle
import os
import numpy as np
from hyperopt import STATUS_OK, STATUS_FAIL
import gc
from hyperopt import space_eval

from elo.elo import StandingsTable


def save_search_space(search_space, initial_best_config=None):
    if initial_best_config is None:
        initial_best_config = [{}]
    pickle.dump(search_space, open("search_space.pkl", "wb"))
    pickle.dump(initial_best_config, open("best_config.pkl", "wb"))
    return search_space, initial_best_config


from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal


@dataclass
class MarlHyperoptConfig:
    file_name: str
    eval_method: Callable
    best_agent: Any
    make_env: Callable
    prep_params: Callable
    agent_class: Any
    agent_config: Callable
    game_config: Callable
    games_per_pair: int
    num_opps: int
    table: Any
    play_game: Callable
    checkpoint_interval: int = 100
    test_interval: int = 100
    test_trials: int = 10
    test_agents: List[str] = field(default_factory=lambda: ["random"])
    device: str = "cpu"


_MARL_CONFIG: MarlHyperoptConfig | None = None


@dataclass
class SarlHyperoptConfig:
    file_name: str
    eval_method: Literal[
        "final_score", "rolling_average", "final_score_rolling_average"
    ]
    make_env: Callable
    prep_params: Callable
    agent_class: Any
    agent_config: Callable
    game_config: Callable
    checkpoint_interval: int = 100
    test_interval: int = 100
    test_trials: int = 10
    last_n_rolling_avg: int = 10  # New parameter for test_score_evaluation 'last_n'
    device: str = "cpu"


_SARL_CONFIG: SarlHyperoptConfig | None = None


def set_sarl_config(config: SarlHyperoptConfig):
    """Set the global SARL config from another file before running hyperopt."""
    global _SARL_CONFIG
    _SARL_CONFIG = config


def set_marl_config(config: MarlHyperoptConfig):
    """Set the global config from another file before running hyperopt."""
    global _MARL_CONFIG
    _MARL_CONFIG = config


def marl_run_training(params, agent_name):
    assert _MARL_CONFIG is not None, "Config not set. Call set_config first."
    config = _MARL_CONFIG

    params = config.prep_params(params)
    try:
        env = config.make_env(render_mode="rgb_array")
    except:
        env = config.make_env()

    agent = config.agent_class(
        env=env,
        config=config.agent_config(
            config_dict=params, game_config=config.game_config(make_env=config.make_env)
        ),
        name=agent_name,
        device=config.device,
        test_agents=config.test_agents,
    )
    agent.checkpoint_interval = config.checkpoint_interval
    agent.test_interval = config.test_interval
    agent.test_trials = config.test_trials

    agent.train()

    if config.eval_method == "elo":
        return elo_evaulation(agent)
    elif config.eval_method == "best_agent_elo":
        return best_agent_elo_evaluation(agent)
    else:
        raise NotImplementedError


def sarl_run_training(params, agent_name):
    """Runs the training for a single-agent configuration and returns the evaluation score."""
    assert _SARL_CONFIG is not None, "SARL Config not set. Call set_sarl_config first."
    config = _SARL_CONFIG

    params = config.prep_params(params)
    try:
        env = config.make_env()  # render_mode="rgb_array"
    except:
        env = config.make_env()

    # Instantiate the agent. Note: SARL agents typically don't need 'test_agents'
    agent = config.agent_class(
        env=env,
        config=config.agent_config(
            config_dict=params, game_config=config.game_config(make_env=config.make_env)
        ),
        name=agent_name,
        device=config.device,
    )
    agent.checkpoint_interval = config.checkpoint_interval
    agent.test_interval = config.test_interval
    agent.test_trials = config.test_trials

    print(f"Starting training for SARL agent: {agent_name}")
    agent.train()
    print("Training complete. Starting test score evaluation.")

    # Evaluate using the test score function
    return test_score_evaluation(
        agent,
        eval_method=config.eval_method,
        num_trials=config.test_trials,
        last_n=config.last_n_rolling_avg,
    )


def marl_objective(params):
    gc.collect()
    assert _MARL_CONFIG is not None, "Config not set. Call set_config first."
    config = _MARL_CONFIG
    print("Params: ", params)
    print("Making environments")
    if os.path.exists(f"./{config.file_name}_trials.p"):
        # it is a normal trial
        trials = pickle.load(open(f"./{config.file_name}_trials.p", "rb"))
        name = "{}_{}".format(config.file_name, len(trials.trials) + 1)
    else:
        # check if it is from the initial best configs
        initial_best_config = pickle.load(open("best_config.pkl", "rb"))
        search_space = pickle.load(open("search_space.pkl", "rb"))
        i = 1
        param_config = config.agent_config(
            config.prep_params(params.copy()),
            config.game_config(make_env=config.make_env),
        )
        for best_config in initial_best_config:
            best_config = space_eval(search_space, best_config)
            if param_config == config.agent_config(
                config.prep_params(best_config),
                config.game_config(make_env=config.make_env),
            ):
                name = "{}_best_{}".format(config.file_name, i)
                print("Using initial best config #{}".format(i))
                break

            i += 1
        else:
            print("No initial best config matched, first trial")
            name = "{}_1".format(config.file_name)
    status = STATUS_OK
    try:
        # add other illegal hyperparameter combinations here
        assert params["min_replay_buffer_size"] >= params["minibatch_size"]
        assert params["replay_buffer_size"] > params["min_replay_buffer_size"]
        # score = run_training([params, env, name])
        score = marl_run_training(params, name)
    except AssertionError as e:
        status = STATUS_FAIL
        print(f"exited due to invalid hyperparameter combination: {e}")
        return {"status": status, "loss": 0}

    print("parallel programs done")
    return {"status": status, "loss": score}  # np.mean(scores_list)


def sarl_objective(params):
    """Hyperopt objective function for Single-Agent Reinforcement Learning (SARL)."""
    gc.collect()
    assert _SARL_CONFIG is not None, "SARL Config not set. Call set_sarl_config first."
    config = _SARL_CONFIG
    print("Params: ", params)
    print("Making environments")

    # --- Naming and Configuration Matching Logic (Reused from MARL Objective) ---
    if os.path.exists(f"./{config.file_name}_trials.p"):
        # it is a normal trial
        trials = pickle.load(open(f"./{config.file_name}_trials.p", "rb"))
        name = "{}_{}".format(config.file_name, len(trials.trials) + 1)
    else:
        # check if it is from the initial best configs
        initial_best_config = pickle.load(open("best_config.pkl", "rb"))
        search_space = pickle.load(open("search_space.pkl", "rb"))
        i = 1
        # Use the config setup for SARL
        param_config = config.agent_config(
            config.prep_params(params.copy()),
            config.game_config(make_env=config.make_env),
        )
        for best_config in initial_best_config:
            best_config = space_eval(search_space, best_config)
            if param_config == config.agent_config(
                config.prep_params(best_config),
                config.game_config(make_env=config.make_env),
            ):
                name = "{}_best_{}".format(config.file_name, i)
                print("Using initial best config #{}".format(i))
                break

            i += 1
        else:
            print("No initial best config matched, first trial")
            name = "{}_1".format(config.file_name)
    # --- End Naming Logic ---

    status = STATUS_OK
    try:
        # Add checks for illegal hyperparameter combinations here (if applicable, e.g., for DQN-like agents)
        # Note: These checks are specific to replay-buffer-based agents, generalize or remove if needed
        assert params["min_replay_buffer_size"] >= params["minibatch_size"]
        assert params["replay_buffer_size"] > params["min_replay_buffer_size"]

        score = sarl_run_training(params, name)
    except AssertionError as e:
        status = STATUS_FAIL
        print(f"exited due to invalid hyperparameter combination: {e}")
        return {"status": status, "loss": 0}
    except Exception as e:
        # Catch other potential training/runtime errors
        status = STATUS_FAIL
        print(f"Training failed with error: {e}")
        return {"status": status, "loss": 0}

    print("parallel programs done")
    # In hyperopt, 'loss' is the value to be minimized. Since score is already negated in
    # sarl_run_training, we return it directly.
    return {"status": status, "loss": score}


def test_score_evaluation(agent, eval_method, num_trials=10, last_n=10):
    if eval_method == "final_score":
        return -agent.test(num_trials=num_trials, dir=f"./checkpoints/")["score"]
    elif eval_method == "rolling_average":
        return -np.around(
            np.mean(
                [
                    stat_dict["score"]
                    for stat_dict in agent.stats["test_score"][-last_n:]
                ]
            ),
            1,
        )
    elif eval_method == "final_score_rolling_average":
        return (
            -agent.test(num_trials=num_trials, dir=f"./checkpoints/")["score"]
            - np.around(
                np.mean(
                    [
                        stat_dict["score"]
                        for stat_dict in agent.stats["test_score"][-last_n:-1]
                    ]
                ),
                1,
            )
        ) / 2


def elo_evaulation(agent):
    assert _MARL_CONFIG is not None, "Config not set. Call set_config first."
    config = _MARL_CONFIG

    opponents_indices = np.random.choice(
        range(len(config.table.players)),
        size=min(config.num_opps, len(config.table.players)),
        replace=False,
    )
    print(
        f"Testing against opponents: {[config.table.players[o].model_name for o in opponents_indices]}"
    )
    config.table.add_player(agent)
    print(config.table.players)

    pickle.dump(config.table, open("hyperopt_elo_table.pkl", "wb"))

    if len(opponents_indices) == 0:
        return 0

    results = config.table.play_matches(
        play_game=config.play_game,
        player_index=len(config.table.players) - 1,
        opponent_indices=opponents_indices,
        games_per_pair=config.games_per_pair,
    )
    pickle.dump(config.table, open("hyperopt_elo_table.pkl", "wb"))

    bayes_elo = config.table.bayes_elo()["Elo table"]
    elo = bayes_elo.iloc[-1]["Elo"]

    print(bayes_elo)
    print(f"Elo: {elo}")
    return -elo


def best_agent_elo_evaluation(agent):
    assert _MARL_CONFIG is not None, "Config not set. Call set_config first."
    config = _MARL_CONFIG
    table = StandingsTable([config.best_agent], start_elo=1400)
    table.add_player(agent)
    results = table.play_matches(
        play_game=config.play_game,
        player_index=1,
        opponent_indices=[0],
        games_per_pair=config.games_per_pair,
    )
    bayes_elo = table.bayes_elo()["Elo table"]
    elo = bayes_elo.iloc[-1]["Elo"]
    best_agent_elo = bayes_elo.iloc[0]["Elo"]
    # get difference between best_agent_elo and 1000
    diff = best_agent_elo - 1400
    elo -= diff

    print(bayes_elo)
    print(f"Elo: {elo}")
    return -elo
