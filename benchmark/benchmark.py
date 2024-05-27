from typing import Callable
import numpy as np
import torch
import torch.utils.benchmark as T
from itertools import product


def mean(m: T.Measurement):
    return np.mean(m.times) * 1e6


def stats(m: T.Measurement) -> list[float]:
    print(m.times[0:5])
    t = np.array(m.raw_times) * 1e6  # nanoseconds
    mean = t.mean()
    max = t.max()
    min = t.min()
    n = len(m.times)
    std = t.std()
    ci = 1.96 * std / (n**0.5)
    return min, max, mean, std, ci


def benchmark_normalize_images(label, sub_label):
    t = T.Timer(
        label="normalize image",
        sub_label=label,
        description=sub_label,
        stmt="y = normalize_images(im)",
        setup="""
from utils import normalize_images
im = torch.randint(0, 255, (4, 84, 84), dtype=torch.float)
""",
    )
    return t


def benchmark_normalize_images_inplace(label, sub_label):
    t = T.Timer(
        label="normalize image (inplace)",
        sub_label=label,
        description=sub_label,
        stmt="y = normalize_images_(im)",
        setup="""
from utils import normalize_images_
im = torch.randint(0, 255, (4, 84, 84), dtype=torch.float)
""",
    )
    return t


def benchmark_make_stack(label, sub_label):
    t = T.Timer(
        label="make stack (inplace)",
        sub_label=label,
        description=sub_label,
        stmt="y = make_stack(im)",
        setup="""
from utils import make_stack
im = torch.randn(20, 60)
""",
    )
    return t


sizes = [1, 4, 20, 64]


def main():
    has_cuda = torch.cuda.is_available()
    print(f"using cuda: {has_cuda}")

    if has_cuda:
        print(f"using device: {torch.cuda.get_device_name(0)}")
    else:
        print(f"using device: {torch.cuda.current_device()}")

    results = []
    timers: list[Callable[[str, str], T.Timer]] = [
        benchmark_normalize_images,
        benchmark_normalize_images_inplace,
        benchmark_make_stack,
    ]
    for a, b in product(sizes, sizes):
        base_label = f"{a} x {b}"
        print(base_label)
        for num_threads in [1, 2, 4, 8]:
            sub_label = f"{num_threads}"
            print(sub_label)
            for t in timers:
                timer = t(base_label, sub_label)
                results.append(timer.blocked_autorange())

    compare = T.Compare(results)
    compare.colorize()
    compare.print()


if __name__ == "__main__":
    main()
