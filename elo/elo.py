# https://github.com/jctops/bayes-elo/tree/master

import numpy as np


def f(D):
    return 1.0 / (1 + np.power(10, D / 400))


def log_likelihood(elos, eloAdvantage, eloDraw, winTable, drawTable):
    l = 0
    for i in range(len(elos)):
        for j in range(len(elos)):
            l += winTable[i][j] * np.log(f(elos[i] - elos[j] - eloAdvantage + eloDraw))
            +0.5 * drawTable[i][j] * np.log(
                1
                - f(elos[i] - elos[j] - eloAdvantage + eloDraw)
                - f(elos[j] - elos[i] + eloAdvantage + eloDraw)
            )
    return l


def get_log_likelihood_for_scipy(winTable, drawTable):
    def l(x):
        elos = x[0:-2]
        eloAdvantage = x[-2]
        eloDraw = x[-1]
        return log_likelihood(elos, eloAdvantage, eloDraw, winTable, drawTable)

    return l


# Global imports
import numpy as np
import pandas as pd
import scipy.optimize


class StandingsTable:
    def __init__(self, players, start_elo=1000):
        self.players = players
        self.start_elo = start_elo
        self.table = pd.DataFrame(
            [[[0, 0, 0] for _ in range(len(players))] for _ in range(len(players))],
            columns=players,
            index=players,
        )
        # for player in players:
        # self.table.loc[player, player] = np.NaN

    def __repr__(self):
        return str(self.table)

    def __str__(self):
        return str(self.table)

    def add_player(self, new_player):
        if new_player not in self.table.index:
            self.players += [new_player]
            self.table[new_player] = [[0, 0, 0] for _ in range(self.table.shape[0])]
            self.table.loc[new_player] = [[0, 0, 0] for _ in range(self.table.shape[1])]
            # self.table.loc[new_player, new_player] = np.NaN
        else:
            print("Player {0} already in table".format(new_player))

    def add_players(self, new_players):
        for new_player in new_players:
            self.add_player(new_player)

    def remove_player(self, player_to_remove):
        if player_to_remove in self.table.index:
            self.table.drop(player_to_remove, axis=0, inplace=True)
            self.table.drop(player_to_remove, axis=1, inplace=True)

    def remove_players(self, players_to_remove):
        for player_to_remove in players_to_remove:
            self.remove_player(player_to_remove)

    def add_result(self, player1, player2, result):
        """
        Result should be +1 for player1 win, 0 for draw, -1 for player2 win
        """
        if player1 != player2:
            if player1 in self.table.columns:
                if player2 in self.table.columns:
                    self.table.loc[player1, player2][1 - result] += 1
                    self.table.loc[player2, player1][1 + result] += 1
                else:
                    print("Player {0} not in table".format(player2))
            else:
                print("Player {0} not in table".format(player1))
        else:
            print("Player cannot play themself")

    def add_results_from_array(self, results_array):
        for row in results_array:
            self.add_result(*row)

    def add_results_from_dataframe(self, results_df):
        for index, row in results_df.iterrows():
            self.add_result(*row)

    def get_win_table(self):
        winTable = np.zeros(self.table.shape)
        for i in range(self.table.shape[0]):
            for j in range(self.table.shape[1]):
                winTable[i][j] = self.table.iloc[i, j][0]
        return winTable

    def get_draw_table(self):
        drawTable = np.zeros(self.table.shape)
        for i in range(self.table.shape[0]):
            for j in range(self.table.shape[1]):
                drawTable[i][j] = self.table.iloc[i, j][1]
        return drawTable

    def calculate_elos(self):
        winTable = self.get_win_table()
        drawTable = self.get_draw_table()
        l = get_log_likelihood_for_scipy(winTable, drawTable)

        x0 = (
            [self.start_elo for _ in range(self.table.shape[0])]
            + [0]
            + [self.start_elo]
        )

        res = scipy.optimize.minimize(l, x0)  # , method='powell')

        return res

    def bayes_elo(self, return_params=True):
        df = pd.DataFrame(
            index=self.players, columns=["Elo", "Games", "Score", "Draws"]
        )

        for player in self.players:
            games = sum([sum(x) for x in self.table.loc[player]])
            df.loc[player, "Games"] = games
            df.loc[player, "Score"] = round(
                sum([x[0] + 0.5 * x[1] for x in self.table.loc[player]]) / games, 3
            )
            df.loc[player, "Draws"] = round(
                sum([x[1] for x in self.table.loc[player]]) / games, 3
            )

        res = self.calculate_elos()
        df["Elo"] = [int(e) for e in res.x[0:-2]]
        eloAdvantage = res.x[-2]
        eloDraw = res.x[-1]

        if return_params:
            return {"Elo table": df, "eloAdvantage": eloAdvantage, "eloDraw": eloDraw}
        else:
            return df

    def play_1v1_tournament(self, players, games_per_pair, play_game):
        tournament_results = []
        for player1 in players:
            results = self.play_matches(player1, players, games_per_pair, play_game)
            tournament_results.extend(results)
        tournament_results = pd.DataFrame(
            tournament_results, columns=["player1", "player2", "result"]
        )
        return tournament_results

    def play_matches(self, player1, players, games_per_pair, play_game):
        results = []
        for opponent in players:
            if opponent != player1:
                for _ in range(games_per_pair // 2):
                    print(
                        f"Playing {player1.__name__} vs {opponent.__name__} game {_+1}"
                    )
                    result = play_game(player1, opponent)
                    results.append((player1.__name__, opponent.__name__, result))

        for opponent in players:
            if opponent != player1:
                for _ in range(games_per_pair // 2):
                    print(
                        f"Playing {opponent.__name__} vs {player1.__name__} game {_+1}"
                    )
                    result = play_game(opponent, player1)
                    results.append(
                        (
                            opponent.__name__,
                            player1.__name__,
                            result,
                        )
                    )
        self.add_results_from_array(results)
        print(self.bayes_elo())
        return results
