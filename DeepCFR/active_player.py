# THIS IS A TEMPLATE FILE TO BE MOVED SOMEWHERE THAT MAKES SENSE ONLY FOR MAKING A CLASS FOR THE MODEL TO KNOW WHO IS THE ACTIVE PLAYER


class ActivePlayer:
    def __init__(self, players):
        """
        Initialize the ActivePlayer class.
        :param players: The number of players in the game.
        """
        self.players = players
        self.active_player = 0

    def set_active_player(self, player):
        """
        Set the active player.
        :param player: The player to set as active.
        """
        if player < 0 or player >= self.players:
            raise ValueError("Invalid player index.")
        self.active_player = player

    def get_active_player(self):
        """
        Get the active player.
        :return: The index of the active player.
        """
        return self.active_player

    def next(self):
        """
        Move to the next player.
        """
        self.active_player = (self.active_player + 1) % self.players
        return self.active_player
