import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from initializers import Initializer, RandomInitializer, BlobInitializer
from vectorhash_functions import difference_of_guassians

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# all hyperaparameters are from the paper unless otherwise stated

### using 64x64 grid for now for faster computation (128x128 in the paper)
grid_size = 64  # 128
###

### hyperparameters for differential equation
tau = 10  # time constant of neural response (ms)
dt = 0.5  # time step for numerical integration (ms)
###

### hyperparameters from the paper for calculating the weight matrix
a = 1  # controls excitatory/inhibitory balance, a = 1 for no excitement
lambda_net = 13  # approximately the periodicity of the formed lattice in the neural sheet
beta = 3 / (lambda_net**2)
gamma = 1.05 * beta
###

### hyperparameters controlling feedforward input (bias)

# "if l=0 and alpha=0 the network generates a static triangular lattice pattern
# ... with overall intensity modulated by the envelope function A"
l = 0
alpha = 0

# "if l, alpha are non-zero, they allow rat velocity (v) to couple to the network
# dynamics and drive a flow of the formed pattern. The magnitues of both l and alpha
# multiplicatively determine how strongly velocity inputs drive the pattern and thus
# control the speed of the flow for a fixed rat speed"

# alpha = 0.10305


### hyperparameters for non-periodic boundary conditions (irrelevant for now)
a_0 = 4
delta_r_ratio = 0.25


class GridCells:
    def __init__(
        self,
        grid_size,
        device,
        tau=tau,
        dt=dt,
        length=l,
        sigma1=gamma,
        sigma2=beta,
        delta_r_ratio=delta_r_ratio,
        alpha=alpha,
        a_0=a_0,
        a=a,
        initializer: Initializer = None,
        name="grid cells",
        debug=False,
    ):
        self.name = name
        self.device = device
        self.debug = debug

        with torch.no_grad():
            if grid_size % 2 != 0:
                raise ValueError("Grid size must be even")

            self.grid_size = grid_size
            self.delta_r = delta_r_ratio * length
            self.tau = tau
            self.dt = dt
            self.length = length
            self.sigma1 = sigma1
            self.sigma2 = sigma2
            self.alpha = alpha
            self.a_0 = a_0
            self.a = a

            if initializer is None:
                initializer = RandomInitializer(
                    grid_size=self.grid_size, device="cuda", mean=1, std=0.5
                )

            self.grid = initializer((grid_size, grid_size)).to(device)
            self.directions_grid = self._generate_directions_grid()
            self.W = self._calc_W0()

            self.grid_history = [self.grid]
            self.times = [0]

            if self.debug:
                print("directions", self.directions_grid)
                print("w", self.W)
                print("grid", self.grid)
                self.plot_directions()
                self.plot_weights()

    def _generate_directions_grid(self):
        if self.debug:
            print("generate directions")

        subgrid = torch.tensor(
            [[[-1, 0], [1, 0]], [[0, -1], [0, 1]]],
            device=self.device,
            dtype=torch.float32,
        )
        g = torch.tile(subgrid, (self.grid_size // 2, self.grid_size // 2, 1))

        # idea: what happens if we shuffle the directions by each 2x2 subgrid?
        # for i in range(0, self.grid_size, 2):
        #     for j in range(0, self.grid_size, 2):
        #         d1, d2, d3, d4 = g[i, j], g[i, j + 1], g[i + 1, j], g[i + 1, j + 1]

        #         directions = torch.stack([d1, d2, d3, d4])
        #         directions = directions[torch.randperm(4)]

        #         g[i, j], g[i, j + 1], g[i + 1, j], g[i + 1, j + 1] = directions

        return g

    def plot_directions(self):
        # label each vector directions with 'N', 'S', 'E', 'W'
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = self.directions_grid[i, j, 0]
                y = self.directions_grid[i, j, 1]

                if x == 0 and y == 1:
                    s = "E"
                elif x == 0 and y == -1:
                    s = "W"
                elif x == 1 and y == 0:
                    s = "S"
                elif x == -1 and y == 0:
                    s = "N"

                plt.text(
                    j,
                    i,
                    s,
                    ha="center",
                    va="center",
                )

        plt.imshow(self.grid.cpu().numpy())

    def _calc_W0(self):
        space = (
            torch.arange(self.grid_size, device=self.device) - self.grid_size // 2
        ).to(torch.float32)
        X, Y = torch.meshgrid(space, space, indexing="ij")

        # equally spaced grid of positions [-n/2, -n/2) x [n/2, n/2)
        self.positions = torch.stack([X, Y], dim=-1)

        # torus topology

        diffs = (
            (
                self.positions.unsqueeze(0).unsqueeze(0)
                - self.positions.unsqueeze(2).unsqueeze(2)
                + self.grid_size // 2
            )
            % self.grid_size
            - self.grid_size // 2
        ) - self.length * self.directions_grid.unsqueeze(2).unsqueeze(2)

        X = difference_of_guassians(
            torch.linalg.vector_norm(diffs, dim=-1), self.a, self.sigma1, self.sigma2
        )  # (grid_size, grid_size, grid_size, grid_size)

        return X

    def step(self, v=None):
        if v is None:
            v = torch.tensor([0, 0], device=self.device, dtype=torch.float32)
        else:
            v = v.to(self.device).to(torch.float32)

        B = self.A(self.positions) * (
            1 + self.alpha * torch.einsum("ijk,k->ij", self.directions_grid, v)
        )
        if self.debug:
            print("B", B)

        Wx = torch.einsum("ijkl,kl->ij", self.W, self.grid)

        if self.debug:
            print("Wx", Wx)

        z = Wx + B
        z = torch.relu(z)

        if self.debug:
            print("Z", z)

        z -= self.grid
        z *= self.dt / self.tau
        self.grid += z
        self.grid_history.append(self.grid)
        self.times.append(self.times[-1] + self.dt)

    def A(self, x):
        return torch.where(
            torch.linalg.vector_norm(x, dim=2) < self.grid_size - self.delta_r,
            1,
            -self.a_0
            * torch.exp(
                (
                    (torch.linalg.vector_norm(x) - self.grid_size + self.delta_r)
                    / self.delta_r
                )
                ** 2
            ),
        )

    def plot_weights(self):
        # plot a subset of the weights, (128, 128, 128, 128) -> (8,8,128,128)

        fig, axs = plt.subplots(8, 8, figsize=(16, 16))

        Is = torch.linspace(0, self.grid_size - 1, 8, dtype=torch.int32)
        Js = torch.linspace(0, self.grid_size - 1, 8, dtype=torch.int32)

        for i in range(8):
            for j in range(8):
                axs[i, j].axis("off")
                a = axs[i, j].imshow(self.W[Is[i], Js[j]].cpu().numpy())
                plt.colorbar(a)

        plt.savefig("weights.png")
        plt.close(fig)

    def animate(self):
        # animate self.grid_history

        fig = plt.figure()

        artists = []

        for i in range(len(self.grid_history)):
            im = plt.imshow(self.grid_history[i].cpu().numpy(), cmap="hot")
            text = plt.text(0, 0, f"time: {self.times[i]} ms")
            artists.append([im, text])

        ani = animation.ArtistAnimation(
            fig, artists, interval=10, blit=True, repeat_delay=1000
        )

        return ani


# flow both the periodic and
# aperiodic network states with unidirectional velocity inputs,
# corresponding to a velocity of 0.8 m/s, in three different directions
# (0,pi/5,pi/2-pi/5) for 250 ms each to heal any strain and defects in
# the formed pattern.


import math

directions = [
    torch.tensor([0.8, 0], device=device, dtype=torch.float32),
    torch.tensor(
        [0.8 * math.cos(torch.pi / 5), 0.8 * math.sin(torch.pi / 5)],
        device=device,
        dtype=torch.float32,
    ),
    torch.tensor(
        [
            0.8 * math.cos(torch.pi / 2 - math.pi / 5),
            0.8 * math.sin(torch.pi / 2 - math.pi / 5),
        ],
        device=device,
        dtype=torch.float32,
    ),
]


# "Initial conditions"
def flow(grid: GridCells):
    # time in ms to flow each direction
    time = 250
    steps = int(time // dt)

    for direction in directions:
        for i in range(steps):
            grid.step(direction)
            print(i)


# experiment 1: no flowing, just 1000 steps
grid = GridCells(grid_size, device)
for i in range(1000):
    grid.step()
    print(i)
anim = grid.animate()
anim.save("experiment1.mp4")
plt.close()

# experiment 2: flow for 250 ms in each direction
grid = GridCells(grid_size, device)
flow(grid)
anim = grid.animate()
anim.save("experiment2.mp4")
plt.close()

# experiment 3: try different initializer
grid = GridCells(grid_size, device, initializer=BlobInitializer(grid_size, device))
for i in range(1000):
    grid.step()
    print(i)
anim = grid.animate()
anim.save("experiment3.mp4")
plt.close()
