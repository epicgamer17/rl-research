import torch


def action_as_plane(num_actions, plane_shape, x):
    onehot_action = torch.zeros(plane_shape).view(-1)
    # print("one hot actions", onehot_action)
    onehot_action[x] = 1
    return onehot_action.view((1,) + plane_shape)


def action_as_onehot(num_actions, plane_shape, x):
    onehot_action = torch.zeros(num_actions)
    onehot_action[x] = 1
    return onehot_action
