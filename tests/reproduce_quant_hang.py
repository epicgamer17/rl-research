import torch
import torch.nn as nn
import torch.ao.quantization
from torch.optim import Adam


def test_hang():
    print("Creating model...")
    m = nn.Linear(10, 10)
    print("Sharing memory...")
    m.share_memory()

    print("Setting config...")
    m.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
    torch.backends.quantized.engine = "qnnpack"

    print("Preparing...")
    torch.ao.quantization.prepare(m, inplace=True)
    print("Prepared")

    print("Creating optimizer...")
    optimizer = Adam(m.parameters(), lr=0.1)
    print("Optimizer created")


if __name__ == "__main__":
    test_hang()
