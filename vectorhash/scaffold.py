import torch

torch.random.manual_seed(0)
from can import ContinousAttractorNetwork
from matplotlib import pyplot as plt
from healers import BurakHealer
import math
from vectorhash_functions import circular_mean, sort_polygon_vertices


def make_can(lambda_net, **kwargs):
    lambda_net = (
        13  # approximately the periodicity of the formed lattice in the neural sheet
    )
    beta = 3 / (lambda_net**2)
    gamma = 1.05 * beta
    return ContinousAttractorNetwork(**kwargs, sigma1=gamma, sigma2=beta)


class GridScaffold:
    @torch.no_grad()
    def __init__(self, lambdas, N_h, sparsity, input_size, device=None, stab_eps=8e-5):
        self.grid_size = 64
        self.lambdas = lambdas
        self.device = device
        self.sparsity = sparsity
        self.input_size = input_size

        self.N_h = N_h
        self.N_g = 0
        self.N_patts = 1

        for l in lambdas:
            self.N_g += l**2
            self.N_patts *= l**2
        k = 2 * len(lambdas)

        self.G = self._G()  # (N_g, N_patts)
        self.W_hg = self._W_hg(mean=0.8 / k, std=1 / (k ** (1 / 2)))  # (N_h, N_g)
        self.H = self.W_hg @ self.G  # (N_h, N_patts)
        self.W_gh = self._W_gh()  # (N_g, N_h)
        self.W_sh = torch.zeros((self.input_size, self.N_h), device=device)
        self.W_hs = torch.zeros((self.N_h, self.input_size), device=device)

        can_args = dict(
            length=1,
            alpha=0.10305,
            device=device,
            grid_size=self.grid_size,
            healer=BurakHealer(
                device=device,
                time=500,
                heal_directions=[
                    [0.8, 0],
                    [0.8 * math.cos(math.pi / 5), 0.8 * math.sin(math.pi / 5)],
                    [
                        0.8 * math.cos(math.pi / 2 - math.pi / 5),
                        0.8 * math.sin(math.pi / 2 - math.pi / 5),
                    ],
                    [0, 0],
                    # [0, 1],
                ],
            ),
        )

        self.stab_eps = stab_eps

        self.cans = [make_can(lambda_net=l, **can_args) for l in lambdas]
        self.stabalize()
        self.boundary_points = self._find_boundaries()
        self.inverses = self._find_inverses(self.boundary_points)
        self.parallelogram_masks = self._find_parallelogram_mask()
        self.g = self._g()
        self.h = torch.zeros(N_h, device=device)

    @torch.no_grad()
    def _G(self):
        G = torch.zeros((self.N_g, self.N_patts), device=self.device)
        height = 0
        for l in self.lambdas:
            G[height : height + l**2, :] = torch.tile(
                torch.eye(l**2, device=self.device), (1, self.N_patts // l**2)
            )
            height += l**2

        return G

    @torch.no_grad()
    def _W_hg(self, mean=0, std=1):
        W_hg = (
            torch.Tensor(self.N_h, self.N_g).normal_(mean=mean, std=std).to(self.device)
        )
        W_hg = (torch.rand_like(W_hg) < self.sparsity).float() * W_hg
        return W_hg

    @torch.no_grad()
    def _W_gh(self):
        return (
            torch.einsum("ix,jx->ijx", self.G, self.H).sum(dim=-1, keepdim=False)
            * 1
            / self.N_h
        )

    @torch.no_grad()
    def _find_clusters(self, i):
        # start at (grid_size // 2, grid_size // 2)
        # do a dfs to find sets of points that are connected
        # find the centers of mass of each set of points
        threshold = 0.25

        pos = torch.tensor([self.grid_size // 2, self.grid_size // 2], device=self.device)
        stack = [pos]
        visited = torch.zeros((self.grid_size, self.grid_size), device=self.device)
        visited[pos[0], pos[1]] = 1

        visited_order = []

        clusters = []

        while len(stack) > 0:
            pos = stack.pop()
            for d in [0, 1], [0, -1], [1, 0], [-1, 0]:
                new_pos = (pos + torch.tensor(d, device=self.device)) % self.grid_size
                if visited[new_pos[0], new_pos[1]] == 0:
                    stack.append(new_pos)
                    visited[new_pos[0], new_pos[1]] = 1
                    visited_order.append(new_pos)

                    if self.cans[i].grid[new_pos[0], new_pos[1]] > threshold:
                        # print("found an interesting point:", new_pos)
                        # find all connected points
                        connected = [new_pos]
                        stack2 = [new_pos]
                        while len(stack2) > 0:
                            pos2 = stack2.pop()
                            for d in (
                                [0, 1],
                                [0, -1],
                                [1, 0],
                                [-1, 0],
                                [0, 2],
                                [0, -2],
                                [2, 0],
                                [-2, 0],
                            ):
                                new_pos2 = (
                                    pos2 + torch.tensor(d, device=self.device)
                                ) % self.grid_size
                                if (
                                    visited[new_pos2[0], new_pos2[1]] == 0
                                    and self.cans[i].grid[new_pos2[0], new_pos2[1]]
                                    > threshold
                                ):
                                    # print(
                                    #     "found a connected interesting point:", new_pos2
                                    # )
                                    stack2.append(new_pos2)
                                    visited[new_pos2[0], new_pos2[1]] = 1
                                    visited_order.append(new_pos2)
                                    connected.append(new_pos2)
                        clusters.append(connected)

        print(len(clusters), "clusters found")
        # find the center of mass of each cluster
        centers = []
        for cluster in clusters:
            positions_tensor = torch.vstack(cluster)  # (c, 2)
            center = circular_mean(positions_tensor, self.grid_size)
            centers.append(center)

        print("centers:", centers)
        return centers, visited_order, clusters

    @torch.no_grad()
    def _find_boundaries(self):
        # for each can:
        # - find the 3 closest centers of mass to the center of the grid
        # - draw lines between the 3 closest centers of mass
        # - shift the lines to have an endpoint at the center of the grid to get points that define a rhombus
        #   when combined with the center of the grid and sorted

        c = torch.tensor([self.grid_size // 2, self.grid_size // 2], device=self.device)
        boundary_points = []

        for i in range(len(self.cans)):
            centers, *_ = self._find_clusters(i)
            # find the 3 closest centers of mass
            closest, indices = torch.topk(
                torch.norm(torch.vstack(centers) - c, dim=-1), 3, largest=False
            )

            # draw a line between the 3 closest centers of mass
            lines = [torch.tensor([0.0, 0.0], device=self.device)]
            for i in range(3):
                for j in range(i + 1, 3):
                    lines.append(centers[indices[i]] - centers[indices[j]])

            # get boundary points that define a rhombus
            points = torch.vstack(lines) + c

            # sort points
            points = sort_polygon_vertices(points)

            boundary_points.append(points)

            print("boundary points:", points)
        return boundary_points

    @torch.no_grad()
    def _find_inverses(self, boundary_points: list[torch.Tensor]):
        inverses = []
        for set in boundary_points:
            # TODO: generalize to more dimensions
            A = torch.vstack([set[1] - set[0], set[3] - set[0]])
            inverses.append(A.inverse())

        return inverses

    @torch.no_grad()
    def _find_parallelogram_mask(self) -> list[torch.Tensor]:
        parallelogram_masks = []
        for i, can in enumerate(self.cans):
            A = self.inverses[i]
            points_in_pgram = torch.all(
                (can.positions.float() @ A >= 0) & (can.positions.float() @ A < 1), dim=-1
            )
            parallelogram_masks.append(points_in_pgram)
        return parallelogram_masks

    @torch.no_grad()
    def _g(self):
        g = torch.zeros(self.N_g, device=self.device)
        c = torch.tensor([self.grid_size // 2, self.grid_size // 2], device=self.device)
        pos = 0
        for i, can in enumerate(self.cans):
            # get most activated cell inside parallelogram relative to the (grid_size // 2, grid_size // 2) vertex of the parallelogram
            mask = self.parallelogram_masks[i]
            masked = can.grid * mask
            max_index = torch.nonzero(masked == torch.max(masked))
            if max_index.shape[0] > 1:
                print("warning: multiple maxes found. picking the first one.")
                max_index = max_index[0]
            max_index = max_index.flatten() - c
            print("max_index:", max_index)
            transformed = (
                self.inverses[i] @ max_index.float()
            ) % 1  # should be in [0, 1]
            print("transformed:", transformed)
            lambda_scaled = torch.floor(transformed * self.lambdas[i]).int()
            onehot = torch.zeros(self.lambdas[i] ** 2, device=self.device)
            onehot[lambda_scaled[0] * self.lambdas[i] + lambda_scaled[1]] = 1
            g[pos : pos + self.lambdas[i] ** 2] = onehot
            pos += self.lambdas[i] ** 2

        self.g = g
        return g

    @torch.no_grad()
    def stabalize(self):
        for j, can in enumerate(self.cans):
            i = 0
            diff = can.grid.clone() - can.step()
            while torch.norm(diff) > self.stab_eps:
                diff = can.grid.clone() - can.step()
                if i % 20 == 0:
                    print(f"[can {j}] stabalizing: step {i}, diff: {torch.norm(diff)}")
                i += 1

    @torch.no_grad()
    def calculate_update(self, input: torch.Tensor, output: torch.Tensor):
        # input: (N)
        # output: (M)
        # M: (M x N)
        ret = (torch.einsum("i,j->ji", input, output)) / (torch.linalg.norm(input) ** 2)
        print(torch.linalg.norm(ret))
        return ret

    @torch.no_grad()
    def store_memory(self, s: torch.Tensor, num_iterations=1):
        """
        Stores sensory input s into the memory model by learning weights.

        :param s: Sensory input vector. - (input_size)
        :param num_iterations: Number of iterations for updating the weights.
        :return: None
        """
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L55C1-L55C69
        for _ in range(num_iterations):
            h = self.W_hg @ self.g

            self.W_gh += self.calculate_update(input=h, output=self.g)
            self.W_sh += self.calculate_update(input=h, output=s)
            self.W_hs += self.calculate_update(input=s, output=h)

    @torch.no_grad()
    def learn_path(self, observations: torch.Tensor, velocities: torch.Tensor):
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L74
        assert observations.shape[0] == velocities.shape[0]
        for i in range(observations.shape[0]):
            print("obs step:", i)
            s = observations[i]
            v = velocities[i]

            for can in self.cans:
                can.step(v)

            print("stabalizing")
            self.stabalize()

            print("storing memory")
            self.store_memory(s)

    @torch.no_grad()
    def learn_path_temporal(self, observations: torch.Tensor):
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L86
        for i in range(observations.shape[0]):
            if i % 10 == 0:
                print(f"learning path temporal: step {i} of {observations.shape[0]}")
            s = observations[i]
            self.store_memory(s, 1)

    @torch.no_grad()
    def recall(self, noisy_observations: torch.Tensor):
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L96
        # noisy_observations: (N, input_size)
        H = noisy_observations @ self.W_hs.T
        G = H @ self.W_gh.T
        denoised_G = self._denoise(G)
        denoised_H = denoised_G @ self.W_hg.T
        denoised_S = denoised_H @ self.W_sh.T

        return denoised_S

    @torch.no_grad()
    def temporal_recall(self, noisy_observations: torch.Tensor):
        # https://github.com/tmir00/TemporalNeuroAI/blob/c37e4d57d0d2d76e949a5f31735f902f4fd2c3c7/model/model.py#L113
        denoised_H = noisy_observations @ self.W_hs.T
        denoised_S = denoised_H @ self.W_sh.T

        return denoised_S

    @torch.no_grad()
    def _denoise(self, G: torch.Tensor):
        if G.ndim == 1:
            G = G.unsqueeze(0)

        denoised_G = torch.zeros_like(G)
        for i in range(G.shape[0]):
            # select the most active pattern
            pos = 0
            for j, l in enumerate(self.lambdas):
                max_index = torch.argmax(G[i, pos : pos + l**2])
                onehot = torch.zeros(l**2, device=self.device)
                onehot[max_index] = 1
                denoised_G[i, pos : pos + l**2] = onehot
                pos += l**2

        return denoised_G


def graph_scaffold(g: GridScaffold):
    # G
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=900)
    ax.imshow(g.G.cpu().numpy(), cmap="gray", aspect="auto")
    fig.savefig("G.png")
    # H
    fig, ax = plt.subplots(1, 1, figsize=(12, 4), dpi=900)
    a = ax.imshow(g.H.cpu().numpy(), cmap="gray", aspect="auto")
    fig.colorbar(a)
    fig.savefig("H.png")
    # W_hg
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=900)
    a = ax.imshow(g.W_hg.cpu().numpy(), cmap="hot")
    fig.colorbar(a)
    fig.savefig("W_hg.png")
    # W_gh
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=900)
    a = ax.imshow(g.W_gh.cpu().numpy(), cmap="hot")
    fig.colorbar(a)
    fig.savefig("W_gh.png")


def visualize_clusters(g: GridScaffold, i=0):
    centers, visited_order, clusters = g._find_clusters(i)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=900)
    a = ax.imshow(g.cans[i].grid.T.cpu().numpy(), cmap="hot")
    # plot clusters
    for cluster in clusters:
        for pos in cluster:
            p = pos.cpu().numpy()
            ax.plot(p[0], p[1], "go")

    # plot centers of mass
    for center in centers:
        c = center.cpu().numpy()
        ax.plot(c[0], c[1], "ro")

    # plot boundary point rhombus
    for point in g.boundary_points[i]:
        p = point.cpu().numpy()
        ax.plot(p[0], p[1], "bo")

    ax.add_patch(
        plt.Polygon(
            g.boundary_points[i].cpu().numpy(),
            fill=False,
            edgecolor="blue",
        )
    )

    fig.colorbar(a)
    fig.savefig(f"grid_{i}.png")

    # plot visited order
    # import matplotlib.animation as animation

    # q = torch.zeros((g.grid_size, g.grid_size), device="cuda")
    # fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=900)
    # artists = []
    # a = ax.imshow(q.cpu().numpy(), cmap="hot")
    # artists.append([a])
    # for pos in visited_order:
    #     q[pos[0], pos[1]] = 1
    #     a = ax.imshow(q.cpu().numpy(), cmap="hot", animated=True)
    #     artists.append([a])

    # ani = animation.ArtistAnimation(fig, artists, interval=100, blit=True)
    # ani.save("visited_order.mp4")


def visualize_parallelogram_points(g: GridScaffold, i=0):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=900)
    a = ax.imshow(g.cans[i].grid.T.cpu().numpy(), cmap="hot")
    points = g.parallelogram_masks[i]
    ax.imshow(points.T.cpu().numpy(), cmap="hot")
    fig.colorbar(a)
    fig.savefig(f"parallelogram_points_{i}.png")


# g = GridScaffold([3, 5], 100, 0.9, device="cuda")
# graph_scaffold(g)
# visualize_clusters(g, 0)
# visualize_clusters(g, 1)
# visualize_parallelogram_points(g, 0)
# visualize_parallelogram_points(g, 1)
# print("g:", g.g)

