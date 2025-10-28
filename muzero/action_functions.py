import torch


def action_as_plane(x):
    onehot_action = torch.zeros((3, 3)).view(-1)
    onehot_action[x] = 1
    return onehot_action.view(1, 3, 3)


def action_as_onehot(x):
    onehot_action = torch.zeros((3, 3)).view(-1)
    onehot_action[x] = 1
    return onehot_action
