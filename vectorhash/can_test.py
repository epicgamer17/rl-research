import torch
import math
from tqdm import tqdm

torch.manual_seed(0)
from can import ContinousAttractorNetwork, plot_can_weights, animate_can

device = "cuda" if torch.cuda.is_available() else "cpu"


def test_static_small():
    can = ContinousAttractorNetwork(
        grid_size=64, device=device, track_history=True, length=0, alpha=0
    )
    plot_can_weights(can, filename="weights_small.png")
    animate_can(can, fps=60, speed=1, filename="can_small.gif")


def test_flowing():
    can = ContinousAttractorNetwork(
        grid_size=64, device=device, track_history=True, length=1, alpha=0.10305
    )
    for i in range(5000):
        can.step(torch.tensor([0.5, 0.1]).to(device))

    for i in range(5000):
        can.step(torch.tensor([0.3, 0.2]).to(device))

    for i in range(5000):
        can.step(torch.tensor([-1.0, 0.0]).to(device))
    animate_can(can, speed=1, fps=30, filename="can_flowing.gif")


def test_circular():
    can = ContinousAttractorNetwork(
        grid_size=64, device=device, track_history=True, length=1, alpha=0.10305
    )
    circles = 2
    steps = 10000

    for i in tqdm(range(steps)):
        phi = 2 * math.pi * i / (steps / circles)
        can.step(0.5 * torch.tensor([math.cos(phi), math.sin(phi)]).to(device))

    animate_can(can, speed=1, fps=30, filename="can_circular.gif")


def test_larger_lambda():
    lambda_net = 20
    beta = 3 / (lambda_net**2)
    gamma = 1.05 * beta
    can = ContinousAttractorNetwork(
        grid_size=64,
        device=device,
        track_history=True,
        length=1,
        alpha=0.10305,
        sigma1=gamma,
        sigma2=beta,
    )
    plot_can_weights(can, filename="weights_larger_lambda.png")
    animate_can(can, speed=1, fps=30, filename="can_larger_lambda.gif")

def test_larger_lambda_flowing():
    lambda_net = 20
    beta = 3 / (lambda_net**2)
    gamma = 1.05 * beta
    can = ContinousAttractorNetwork(
        grid_size=64,
        device=device,
        track_history=True,
        length=1,
        alpha=0.10305,
        sigma1=gamma,
        sigma2=beta,
    )
    for i in range(5000):
        can.step(torch.tensor([0.5, 0.1]).to(device))

    for i in range(5000):
        can.step(torch.tensor([0.3, 0.2]).to(device))

    for i in range(5000):
        can.step(torch.tensor([-1.0, 0.0]).to(device))
    animate_can(can, speed=1, fps=30, filename="can_larger_lambda_flowing.gif")

def test_larger_lambda_circular():
    lambda_net = 20
    beta = 3 / (lambda_net**2)
    gamma = 1.05 * beta
    can = ContinousAttractorNetwork(
        grid_size=64,
        device=device,
        track_history=True,
        length=1,
        alpha=0.10305,
        sigma1=gamma,
        sigma2=beta,
    )
    circles = 2
    steps = 10000

    for i in tqdm(range(steps)):
        phi = 2 * math.pi * i / (steps / circles)
        can.step(0.5 * torch.tensor([math.cos(phi), math.sin(phi)]).to(device))

    animate_can(can, speed=1, fps=30, filename="can_larger_lambda_circular.gif")

test_static_small()
test_flowing()
test_circular()
test_larger_lambda()
test_larger_lambda_flowing()
test_larger_lambda_circular()