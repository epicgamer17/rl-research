class GameConfig:
    def __init__(self):
        # determine if it is:
        # 1. discrete or conitnuous
        # 2. if discrete could do num actions
        # 3. observation shape
        # 4. if the observation is an image
        # 5. if it is deterministic
        # 6. number of players (might not need this idk) <- it would likely be for muzero but could also be for rainbow and stuff when they play multiplayer games (like connect 4)
        # 7. if it has legal moves info? <- could do this instead of the weird check i am doing for alphazero right now
        pass
