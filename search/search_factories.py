from search.algorithms import GumbelSequentialHalving, SearchAlgorithm, UCTSearch


def create_mcts(config, device, num_actions) -> SearchAlgorithm:
    if config.gumbel:
        return GumbelSequentialHalving(config, device, num_actions)
    else:
        return UCTSearch(config, device, num_actions)
