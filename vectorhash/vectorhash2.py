import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from healers import DefaultHealer

from initializers import Initializer, RandomInitializer
from vectorhash_functions import difference_of_guassians

torch.manual_seed(0)
import random

random.seed(0)

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

l = 1
alpha = 0.10305


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
        healer=None,
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

            if healer is None:
                healer = DefaultHealer()

            self.grid = initializer((grid_size, grid_size)).to(device)
            self.W = self._calc_W0()
            self.A = self._A(self.positions)

            if self.debug:
                print("w", self.W)
                print("grid", self.grid)
            self.plot_weights()

            self.grid_history = [self.grid.cpu().numpy()]
            self.times = [0]
            healer(self)

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
        directions_grid = torch.tile(
            torch.tensor(
                [[[0, 1], [0, -1]], [[1, 0], [-1, 0]]],
                device=self.device,
                dtype=torch.float32,
            ),
            (self.grid_size // 2, self.grid_size // 2, 1),
        )

        # torus topology

        diffs = (
            (
                self.positions.unsqueeze(0).unsqueeze(0)
                - self.positions.unsqueeze(2).unsqueeze(2)
                + self.grid_size // 2
            )
            % self.grid_size
            - self.grid_size // 2
            - self.length * directions_grid
            + self.grid_size // 2
        ) % self.grid_size - self.grid_size // 2

        X = difference_of_guassians(
            torch.linalg.vector_norm(diffs, dim=-1), self.a, self.sigma1, self.sigma2
        )  # (grid_size, grid_size, grid_size, grid_size)

        return X

    def _calc_b(self, v):
        dotted = (
            torch.tensor(
                [[-1, 0], [1, 0], [0, -1], [0, 1]],
                device=self.device,
                dtype=torch.float32,
            )
            @ v
        )
        subgrid = dotted.reshape(2, 2)
        tiled = torch.tile(subgrid, (self.grid_size // 2, self.grid_size // 2))
        return tiled

    def step(self, v=None):
        if v is None:
            v = torch.tensor([0, 0], device=self.device, dtype=torch.float32)
        else:
            v = v.to(self.device).to(torch.float32)

        if (self.alpha * torch.norm(v)) > 1:
            print("warning: alpha * |v| > 1 will cause instability.", f"v={v}")

        Wx = torch.einsum("ijkl,kl->ij", self.W, self.grid)
        B = self.A * (1 + self.alpha * self._calc_b(v))

        if self.debug:
            print("B", B)
            print("Wx", Wx)

        z = Wx + B
        if self.debug:
            print("Wx+B", z)
        z = torch.relu(z)

        z = (z - self.grid) * self.dt / self.tau
        self.grid += z
        self.grid_history.append(self.grid.cpu().numpy())
        self.times.append(self.times[-1] + self.dt)

    def _A(self, x):
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

    def animate(self, fps=None, speed=1):
        if fps is None:
            fps = 1000 * speed / self.dt

        if fps > 1000 * speed / self.dt:
            print(
                "fps should be less than 1000 * speed / dt (or else the same state will be rendered multiple times)"
            )

        fig = plt.figure(figsize=(8, 9))

        artists = []

        im = plt.imshow(self.grid_history[0], cmap="hot")
        text = plt.text(0, -0.5, f"time: {self.times[0]} ms")
        artists.append([im, text])

        for i in range(1, len(self.grid_history), int(1000 * speed // (fps * grid.dt))):
            im = plt.imshow(self.grid_history[i], cmap="hot", animated=True)
            text = plt.text(0, -0.5, f"time: {self.times[i]} ms")
            artists.append([im, text])

        ani = animation.ArtistAnimation(
            fig,
            artists,
            interval=1000 * speed / fps,
            blit=True,
            repeat_delay=1000,
        )

        return ani


# animation 1: flow for 250 ms in each direction (default described in paper), l=0, alpha=0
grid = GridCells(grid_size, device, length=0, alpha=0)
anim = grid.animate(fps=30, speed=0.1)
anim.save("anim1.gif")
plt.close()

# animation 2: flow for 250 ms in each direction, then flow for 1000 east at 1 m/s, l=1, alpha=0.10305
grid = GridCells(grid_size, device, length=1, alpha=0.10305)

for i in range(5000):
    grid.step(torch.tensor([0.1, 0.1], device=device, dtype=torch.float32))
    print(i)

anim = grid.animate(fps=30)
anim.save("anim2.gif")
plt.close()

