import torch


# Assuming the use of torch.nn.functional.one_hot is available in the environment
# For flexibility, we use torch.zeros and scatter_
def action_as_plane(num_actions, plane_shape, x):
    """
    Converts a batch of actions (indices) into a batch of one-hot planes.

    :param num_actions: Total number of possible actions.
    :param plane_shape: The (H, W) or (C, H, W) shape of the plane.
    :param x: A tensor of action indices, shape (B,).
    :return: A tensor of one-hot action planes, shape (B, C_plane, H, W) or similar.
    """
    B = x.shape[0]  # Batch size

    # 1. Create a 1D one-hot vector for the batch (B, num_actions)
    onehot_action_1d = torch.zeros(
        (B, num_actions), dtype=torch.float32, device=x.device
    )
    # Scatter the value 1 into the specified indices (x) along dimension 1
    # x needs to be (B, 1) to scatter correctly into (B, num_actions)
    onehot_action_1d.scatter_(1, x.unsqueeze(-1).long(), 1.0)

    # 2. Reshape the one-hot vectors to match the spatial plane shape for concatenation
    # plane_shape could be (C, H, W) or (H, W). We need the total size.
    target_plane_size = torch.Size((B,) + plane_shape)

    # We flatten the target plane size, then reshape the one-hot actions to match
    # Note: If the product of plane_shape elements is not equal to num_actions,
    # the original function logic is flawed, but assuming they match:
    # onehot_action_1d should be (B, num_actions)

    # Assuming plane_shape is the shape of the *single* plane (H, W) or (C, H, W)
    # and that num_actions is the total size of this plane flattened.
    return onehot_action_1d.view(target_plane_size)


def action_as_onehot(num_actions, plane_shape, x):
    """
    Converts a batch of actions (indices) into a batch of one-hot vectors.

    :param num_actions: Total number of possible actions.
    :param plane_shape: (Unused in this function, kept for signature consistency)
    :param x: A tensor of action indices, shape (B,).
    :return: A tensor of one-hot vectors, shape (B, num_actions).
    """
    B = x.shape[0]  # Batch size

    # Use torch.nn.functional.one_hot (or scatter_) for clean batch one-hot encoding
    onehot_action = torch.zeros((B, num_actions), dtype=torch.float32, device=x.device)
    # Scatter the value 1 into the specified indices (x) along dimension 1
    # x needs to be (B, 1) to scatter correctly into (B, num_actions)
    onehot_action.scatter_(1, x.unsqueeze(-1).long(), 1.0)

    return onehot_action
