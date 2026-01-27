import torch
import torch.nn as nn
import torch.ao.quantization


def test_lstm_args():
    torch.backends.quantized.engine = "qnnpack"

    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(10, 20)

    m = Container()
    q_m = torch.ao.quantization.quantize_dynamic(m, {nn.LSTM}, dtype=torch.qint8)

    lstm = q_m.lstm

    # Try calling without args to see error
    try:
        lstm.set_weight_bias()
    except Exception as e:
        print(f"Error (no args): {e}")


if __name__ == "__main__":
    test_lstm_args()
