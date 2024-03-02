class FastSumTree(object):
    # https://medium.com/free-code-camp/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682

    def __init__(self, capacity):
        self.capacity = int(
            capacity
        )  # number of leaf nodes (final nodes) that contains experiences

        self.tree = np.zeros(2 * self.capacity - 1)  # sub tree
        # self.data = np.zeros(self.capacity, object)  # contains the experiences

    def add(self, idx: int, val: float):
        """Set value in tree."""
        tree_index = idx + self.capacity - 1
        # self.data[self.data_pointer] = data
        self.update(tree_index, val)

    def __getitem__(self, idx: int) -> float:
        """Get real value in leaf node of tree."""
        assert 0 <= idx < self.capacity

        return self.tree[self.capacity + idx]

    def update(self, tree_index, val):
        change = val - self.tree[tree_index]
        # print("change", change)
        self.tree[tree_index] = val
        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
            # print("new value", self.tree[tree_index])


    def retrieve(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        return leaf_index, self.tree[leaf_index]

    @property
    def total_priority(self):
        return self.tree[0]