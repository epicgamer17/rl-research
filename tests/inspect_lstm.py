import torch
import torch.nn as nn
import torch.ao.quantization


def inspect_lstm():
    torch.backends.quantized.engine = "qnnpack"

    # Use a container to ensure correct quantization behavior
    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(10, 20)

    m = Container()
    q_m = torch.ao.quantization.quantize_dynamic(m, {nn.LSTM}, dtype=torch.qint8)

    lstm = q_m.lstm
    print(f"Type: {type(lstm)}")

    # Check for likely setters
    for attr in dir(lstm):
        if "set" in attr and "weight" in attr:
            print(f"Found candidate: {attr}")
            try:
                method = getattr(lstm, attr)
                # Print docstring or help
                print(f"Doc: {method.__doc__}")
            except:
                pass


if __name__ == "__main__":
    inspect_lstm()
