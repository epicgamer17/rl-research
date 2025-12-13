from search.modular_search import SequentialHalvingSearch, SearchAlgorithm, UCTSearch


def create_mcts(config, device, num_actions) -> SearchAlgorithm:
    if config.gumbel:
        return SequentialHalvingSearch(config, device, num_actions)
    else:
        return UCTSearch(config, device, num_actions)
