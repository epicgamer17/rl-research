# TODO: some of these are less "features" expansion is not a "feature". this organization is nice, but actually we will end up having multiple "features" in the same file, like puct and gumbel scoring etc. solution may be to make phases folders, and have the files be features again.

# TODO: basically make this a pytorch port of mctx. Leave it at that, maybe change the api a little to match the library, but functionally, and function wise just port it.

# TODO:URGENT/MAJOR: Don't force cananonical values in MCTS

# TODO: should this also be used for dreamer? if I do that, am I making this too generic (if that is possible). Conceptually i am torn. its nice to bridge the concepts, but at a high level they are different (one is about training in the model, the other is searching), but can be seen as similar if you consider learning between steps as a form of "search". The question i guess arises, how should other search types like AlphaBeta search, or A-star or CFR etc fit into this framework? Will I be "expanding" the single search to be more and more generic? or will i instead just create new folders and functions for each type? If i choose to create new functions, where is the line between a new function, and expanding capabilities of an existing one (for example i would say that adding nplayer and stochastic could be seen as extensions on MCTS). Its likely the case extentions and composition should be used to prevent a combinatorial explosion of classes. Is it possible to make a "generic" search system that covers search in general, and then specializes for each algorithm by composition of functions?

from .backpropagation import backpropagate_
from .expansion import expand_node_
from .policies import get_mcts_visit_policy
from .fpu import (
    FPUFn,
    fpu_zero,
    fpu_constant,
    fpu_parent_value,
    fpu_mixed_value,
    fpu_min_value,
)
from .qtransforms import (
    get_qvalues,
    get_completed_qvalues,
    qtransform_completed_by_mix_value,
    qtransform_by_parent_and_siblings,
    qtransform_by_min_max,
    qtransform_identity,
)
from .search import mcts_search

from .selection import (
    make_stochastic_selection_fn,
    puct_score,
    puct_selection,
    quasi_sampling_selection,
    select_leaf,
)
from .tree import init_mcts_tree

"""
Our MCTS implementation is essenitally a rewrite of mctx, https://github.com/google-deepmind/mctx but in PyTorch. We have however added additional features, with more planned in the future. Those include n-player game functionality and sampled muzero functionality.
"""
