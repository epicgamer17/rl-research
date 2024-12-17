import torch
from initializers import Initializer, RandomInitializer
from healers import Healer, BurakHealer
from vectorhash_functions import difference_of_guassians

import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


class ContinousAttractorNetwork:
    @torch.no_grad()
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
        healer: Healer = None,
        track_history=False,
    ):
        if initializer == None:
            self.initializer = RandomInitializer(device=device)
        if healer == None:
            healer = BurakHealer(device=device)
        self.grid_size = grid_size
        self.device = device
        self.tau = tau
        self.dt = dt
        self.length = length
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.delta_r_ratio = delta_r_ratio
        self.alpha = alpha
        self.a_0 = a_0
        self.a = a
        self.name = name
        self.delta_r = delta_r_ratio * length
        self.grid = self.initializer((grid_size, grid_size))
        self.preferred_directions = (
            torch.tensor([[[0, 1], [0, -1]], [[1, 0], [-1, 0]]]).float().to(device)
        )
        self.W = self._W()
        self.A = self._A(self.positions)
        self.track_history = track_history
        if self.track_history:
            self.grid_history = [self.grid.cpu().numpy()]
            self.time_history = [0]
            self.v_history = [0]
        healer(self)

    @torch.no_grad()
    def _W(self):
        space = (
            torch.arange(self.grid_size, device=self.device).float() - self.grid_size // 2
        )
        X, Y = torch.meshgrid(space, space, indexing="ij")
        self.positions = torch.stack([X, Y], dim=-1)
        directions_grid = torch.tile(
            self.preferred_directions, (self.grid_size // 2, self.grid_size // 2, 1)
        )
        # diffs = (
        #     self.positions.unsqueeze(2).unsqueeze(2)
        #     - self.positions.unsqueeze(0).unsqueeze(0)
        #     - self.length * directions_grid
        # ) % self.grid_size - self.grid_size // 2
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
        norms = torch.norm(diffs, dim=-1)
        return difference_of_guassians(norms, self.a, self.sigma1, self.sigma2)

    @torch.no_grad()
    def _B(self, v: torch.Tensor):
        tile = torch.einsum("ijk,k->ij", self.preferred_directions, v)
        tiled = torch.tile(tile, (self.grid_size // 2, self.grid_size // 2))
        return self.A * (1 + self.alpha * tiled)

    @torch.no_grad()
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

    @torch.no_grad()
    def step(self, v=None):
        if v == None:
            v = torch.tensor([0, 0]).to(self.device).to(torch.float32)
        if (self.alpha * torch.norm(v)) > 1:
            print("warning: alpha * |v| > 1 will cause instability.", f"v={v}")
        Wx = torch.einsum("ijkl,kl->ij", self.W, self.grid)
        B = self._B(v)
        z = torch.relu(Wx + B)
        self.grid = (1.0 - self.dt / self.tau) * self.grid + self.dt / self.tau * z
        if self.track_history:
            self.grid_history.append(self.grid.cpu().numpy())
            self.time_history.append(self.time_history[-1] + self.dt)
            self.v_history.append(v.cpu().numpy())
        return self.grid


import math


def plot_can_weights(can, positions=None, filename="weights.png"):
    if positions is None:
        space = (
            torch.linspace(0, can.grid_size - 1, 8, dtype=torch.int32)
            - can.grid_size // 2
        )
        X, Y = torch.meshgrid(space, space, indexing="ij")
        positions = torch.stack([X, Y], dim=-1).flatten(0, 1)
    l1 = int(math.sqrt(len(positions)))
    if l1**2 == len(positions):
        l2 = l1
    else:
        l2 = l1 + 1
    fig, axs = plt.subplots(l1, l2, figsize=(18, 16))
    ticks = torch.arange(0, can.grid_size - 1, can.grid_size // 4)
    labels = ticks - can.grid_size // 2
    ticks = ticks.tolist()
    labels = labels.tolist()
    for i, pos in enumerate(positions):
        x, y = pos
        a = axs[i // l2, i % l2].imshow(can.W[x, y].cpu().numpy())
        axs[i // l2, i % l2].set_xticks(ticks, labels)
        axs[i // l2, i % l2].set_yticks(ticks, labels)
        axs[i // l2, i % l2].xaxis.tick_top()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(a, cax=cbar_ax)
    plt.savefig(filename)
    plt.close(fig)


def animate_can(can: ContinousAttractorNetwork, fps=None, speed=1, filename=None):
    if can.grid_history is None:
        raise ValueError("track_history must be set to True to animate")
    dt = can.dt
    if fps is None:
        fps = 1000 * speed / dt
    if fps > 1000 * speed / dt:
        print(
            "fps should be less than 1000 * speed / dt (or else the same state will be rendered multiple times)"
        )
    interval = 1000 * speed / fps
    history_step = int(interval // dt)
    total_frames = (len(can.grid_history) - 1) // history_step + 1
    print("interval:", interval, "history_step:", history_step, "total_frames:", total_frames, "fps:", fps, "speed:", speed)
    fig = plt.figure(figsize=(8, 9))
    artists = []
    im = plt.imshow(can.grid_history[0], cmap="hot")
    text = plt.text(0, -0.5, f"time: {can.time_history[0]} ms")
    artists.append([im, text])
    for i in range(1, len(can.grid_history), history_step):
        im = plt.imshow(can.grid_history[i], cmap="hot", animated=True)
        s = f"time: {can.time_history[i]} ms | v: {can.v_history[i]}"
        text = plt.text(0, -0.5, s, animated=True)
        artists.append([im, text])
    ani = animation.ArtistAnimation(
        fig,
        artists,
        interval=interval,
        blit=True,
        repeat_delay=1000,
    )
    ani.save(
        filename,
        progress_callback=lambda i, n: print(f"writing frame {i} of {total_frames}"),
    )
    return ani
