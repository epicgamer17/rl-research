import random
import bitboard as bb
import time
import typing

# based on http://mediocrechess.blogspot.com/2007/01/guide-transposition-tables.html

flags = {
    "EXACT": 0,
    "LOWERBOUND": 1,
    "UPPERBOUND": 2,
}


class Hashentry:
    def __init__(self, state, depth, eval, flag, best_move, ancient):
        self.state = state
        self.depth = depth
        self.eval = eval
        self.flag = flag
        self.best_move = best_move
        self.ancient = ancient

    def __str__(self):
        return f"State: {self.state}, Value: {self.eval}, Depth: {self.depth}, Flag: {self.flag}"

    def __repr__(self):
        return f"State: {self.state}, Value: {self.eval}, Depth: {self.depth}, Flag: {self.flag}"


class TranspositionTable:
    replacement_strategies = {
        "ALWAYS_REPLACE": 0,
        "REPLACE_BY_DEPTH": 1,
        # The idea is that an entry with a greater depth used more time to get to the evaluation, hence keeping the entry with the greater depth will save more time in the future (Mediocre Chess Blog, 2007)
        "REPLACE_BY_DEPTH_AND_ALWAYS": 2,  # not implemented yet
    }

    def __init__(
        self,
        buckets=4,
        bucket_size=1048583,  # 2^20 + 7 is a prime number
        replacement_strategy=replacement_strategies["REPLACE_BY_DEPTH"],
    ):
        self.size = bucket_size
        if (
            replacement_strategy
            == self.replacement_strategies["REPLACE_BY_DEPTH_AND_ALWAYS"]
        ):
            raise NotImplementedError("REPLACE_BY_DEPTH_AND_ALWAYS not implemented yet")

        self.replacement_strategy = replacement_strategy
        self.num_buckets = buckets
        self.buckets: typing.List[typing.Dict[int, Hashentry]] = list()
        for i in range(buckets):
            self.buckets.append(dict())

    def _store_always_replace(self, state, depth, eval, flag, best_move, ancient):
        key = state % self.size
        for i in range(self.num_buckets):
            if key not in self.buckets[i]:
                self.buckets[i][key] = Hashentry(
                    state=state,
                    depth=depth,
                    eval=eval,
                    flag=flag,
                    best_move=best_move,
                    ancient=ancient,
                )
                return

        random_bucket = random.randint(0, self.num_buckets - 1)
        self.buckets[random_bucket][key] = Hashentry(
            state=state,
            depth=depth,
            eval=eval,
            flag=flag,
            best_move=best_move,
            ancient=ancient,
        )

    def _store_replace_by_depth(
        self, state: int, depth, eval, flag, best_move, ancient
    ):
        key = state % self.size
        min_depth = 44
        min_bucket = 0
        for i in range(self.num_buckets):
            if key not in self.buckets[i] or self.buckets[i][key].depth < depth:
                self.buckets[i][key] = Hashentry(
                    state=state,
                    depth=depth,
                    eval=eval,
                    flag=flag,
                    best_move=best_move,
                    ancient=ancient,
                )
                return

            if self.buckets[i][key].depth < min_depth:
                min_depth = self.buckets[i][key].depth
                min_bucket = i

        self.buckets[min_bucket][key] = Hashentry(
            state=state,
            depth=depth,
            eval=eval,
            flag=flag,
            best_move=best_move,
            ancient=ancient,
        )

    def _store_replace_by_depth_and_always(
        self, state, depth, eval, flag, best_move, ancient
    ):
        return None
        # not implemented yet - need to store two entries for each state

    def store(self, state, depth, eval, flag, best_move, ancient):
        if self.replacement_strategy == self.replacement_strategies["ALWAYS_REPLACE"]:
            self._store_always_replace(state, depth, eval, flag, best_move, ancient)
        elif (
            self.replacement_strategy == self.replacement_strategies["REPLACE_BY_DEPTH"]
        ):
            self._store_replace_by_depth(state, depth, eval, flag, best_move, ancient)
        elif (
            self.replacement_strategy
            == self.replacement_strategies["REPLACE_BY_DEPTH_AND_ALWAYS"]
        ):
            self._store_replace_by_depth_and_always(
                state, depth, eval, flag, best_move, ancient
            )

    def lookup(self, state):
        key = state % self.size
        for i in range(self.num_buckets):
            if key in self.buckets[i]:
                entry = self.buckets[i][key]
                if entry.state == state:
                    return entry

        return None

    def clear(self):
        for i in range(self.num_buckets):
            self.buckets[i].clear()


class Search:
    def __init__(
        self,
        scoring_function=None,
        max_depth=5,
        transposition_table=TranspositionTable(),
        max_time=5,
        debug=False,
    ):
        self.max_depth = max_depth
        self.transposition_table = transposition_table
        self.max_time = max_time
        self.debug = debug
        self.scoring_function = scoring_function

    def _max_value(
        self, state: bb.Bitboard, turn: int, depth: int, max_depth: int, alpha, beta
    ):
        hash = state.hash()
        if self.debug:
            print(
                f"Max Value:\n {state}, Turn: {turn}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}"
            )
        # check if position has been evaluated before
        memorized = self.transposition_table.lookup(hash)
        if memorized is not None and memorized.depth >= max_depth - depth:
            if memorized.flag == flags["EXACT"]:
                if self.debug:
                    print(f"Memorized Exact: {memorized}")
                return memorized.eval, memorized.best_move

            # if the best value of the minimizer (beta) < the memorized value of this position, then we can prune as the minimizer will never allow this position to be reached
            if memorized.flag == flags["LOWERBOUND"] and beta <= memorized.eval:  #
                if self.debug:
                    print(f"Memorized Lowerbound: {memorized}")
                return memorized.eval, memorized.best_move

        # check if position is a terminal node
        if state.check_victory()[0] != 0:
            self.transposition_table.store(
                hash, depth, -43 + depth, flags["EXACT"], 0, 0
            )

            if self.debug:
                print(f"Terminal Node found: Depth: {depth}, Value: {-43 + depth}")
            return -43 + depth, 0

        if state.generate_moves() == []:
            self.transposition_table.store(hash, depth, 0, flags["EXACT"], 0, 0)

            if self.debug:
                print(f"Terminal Node found: Depth: {depth}, Value: 0")
            return 0, 0

        if depth == max_depth:
            eval, best_move = self.scoring_function(state, turn)
            self.transposition_table.store(
                hash, depth, eval, flags["EXACT"], best_move, 0
            )

            if self.debug:
                print(
                    f"Scoring Function: Best move: {best_move}, Depth: {depth}, Value: {eval}"
                )
            return eval, best_move

        # if position has not been evaluated before, evaluate it
        value = float("-inf")
        move = 3
        moves = state.generate_moves()

        if self.debug:
            print(f"Max value moves: {moves}")

        for move in moves:  # TODO - different move ordering
            if self.debug:
                print(f"Max value plays: {move}")

            state.move(turn % 2, move)
            move_value, best_move = self._min_value(
                state, turn + 1, depth + 1, max_depth, alpha, beta
            )
            state.unmove(turn % 2, move)

            if move_value > value:
                value = move_value
                move = best_move

            # if the best value of the minimizer (beta) < the value of this position, then we can prune as the minimizer will never allow this position to be reached
            if value >= beta:
                self.transposition_table.store(
                    hash, depth, value, flags["LOWERBOUND"], move, 0
                )

                if self.debug:
                    print(f"Pruned: Depth: {depth}, Value: {value}, Move: {move}")

                return value, move
            if value > alpha:
                alpha = value

        if self.debug:
            print(
                f"Max Value ret: Turn: {turn}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}, Value: {value}, Move: {move}"
            )
        return value, move

    def _min_value(
        self, state: bb.Bitboard, turn: int, depth: int, max_depth: int, alpha, beta
    ):
        hash = state.hash()
        if self.debug:
            print(
                f"Min Value:\n {state}, Turn: {turn}, Depth: {depth}, Alpha: {alpha}, Beta: {beta}"
            )
        # check if position has been evaluated before
        memorized = self.transposition_table.lookup(hash)
        if memorized is not None and memorized.depth >= max_depth - depth:
            if memorized.flag == flags["EXACT"]:
                if self.debug:
                    print(f"Memorized Exact: {memorized}")
                return memorized.eval, memorized.best_move

            # if the best value of the maximizer (alpha) > the memorized value of this position, then we can prune as the maximizer will never go to this position
            if memorized.flag == flags["UPPERBOUND"] and alpha >= memorized.eval:
                if self.debug:
                    print(f"Memorized Upperbound: {memorized}")
                return memorized.eval, memorized.best_move

        # check if position is a terminal node
        if state.check_victory()[0] != 0:
            if self.debug:
                print(f"Terminal Node found: Depth: {depth}, Value: {43 - depth}")

            self.transposition_table.store(
                hash, depth, 43 - depth, flags["EXACT"], 0, 0
            )
            return 43 - depth, 0

        if state.generate_moves() == []:
            self.transposition_table.store(hash, depth, 0, flags["EXACT"], 0, 0)

            if self.debug:
                print(f"Terminal Node found: Depth: {depth}, Value: 0")
            return 0, 0

        if depth == max_depth:
            eval, best_move = self.scoring_function(state, turn)
            self.transposition_table.store(
                hash, depth, eval, flags["EXACT"], best_move, 0
            )

            if self.debug:
                print(
                    f"Scoring Function: Best move: {best_move}, Depth: {depth}, Value: {eval}"
                )
            return eval, best_move

        # if position has not been evaluated before, evaluate it
        moves = state.generate_moves()
        value = float("inf")
        move = 3

        if self.debug:
            print(f"Min value moves: {moves}")

        for move in moves:  # TODO - different move ordering
            if self.debug:
                print(f"Min value plays: {move}")

            state.move(turn % 2, move)
            move_value, best_move = self._max_value(
                state, turn + 1, depth + 1, max_depth, alpha, beta
            )
            state.unmove(turn % 2, move)

            if move_value < value:
                value = move_value
                best_move = move

            # if the best value of the maximizer (alpha) > the value of this position, then we can prune as the maximizer will never go to this position
            if value <= alpha:
                if self.debug:
                    print(f"Pruned: Depth: {depth}, Value: {value}, Move: {move}")
                self.transposition_table.store(
                    hash, depth, value, flags["UPPERBOUND"], move, 0
                )
                return value, move
            if value < beta:
                beta = value

        if self.debug:
            print(
                f"Min Value ret: Alpha: {alpha}, Beta: {beta}, Value: {value}, Move: {move}"
            )
        return value, move

    def alpha_beta_from_root(
        self,
        state: bb.Bitboard,
        turn: int,
        depth: int = 0,
        max_depth: int = 2,
        alpha=float("-inf"),
        beta=float("inf"),
    ):
        if turn % 2 == 0:
            return self._max_value(state, turn, depth, max_depth, alpha, beta)

        return self._min_value(state, turn, depth, max_depth, alpha, beta)

    def iterative_deepening(self, state: bb.Bitboard, turn: int, max_depth: int):
        best_move = 3
        time_start = time.time()
        time_step = time_start
        run_max_depth = max_depth

        if run_max_depth == None:
            run_max_depth = self.max_depth

        if self.debug:
            print(f"Searching depth: {1}")

        value, best_move = self.alpha_beta_from_root(state, turn, max_depth=1)

        for depth in range(2, run_max_depth + 1):
            if self.debug:
                print(f"Searching depth: {depth}")

            if time.time() - time_start > self.max_time:
                if self.debug:
                    print(f"Time limit reached at depth {depth}")
                break

            value, best_move = self.alpha_beta_from_root(state, turn, max_depth=depth)

            time_taken = time.time() - time_step
            time_step = time.time()

        if self.debug:
            print(
                f"Board: {bb.Bitboard}, Depth: {depth}, Value: {value}, Best Move: {best_move}, Time taken: {time_taken}"
            )

        return value, best_move


# def test_search():
#     search = Search(
#
#         max_depth=5,
#         transposition_table=TranspositionTable(
#             buckets=4,
#             bucket_size=1048583,
#             replacement_strategy=TranspositionTable.replacement_strategies[
#                 "REPLACE_BY_DEPTH"
#             ],
#         ),
#         max_time=5,
#     )
#     state = bb.Bitboard()
#     best_move = search.iterative_deepening(state, 0)
#     print(f"Best Move: {best_move}")
#
#
# test_search()
#
