import torch
import torch.nn as nn
import torch.ao.quantization


def test_lstm_dict():
    torch.backends.quantized.engine = "qnnpack"

    class Container(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(10, 20)

    m = Container()
    # Set known weights
    with torch.no_grad():
        m.lstm.weight_ih_l0.fill_(1.0)
        m.lstm.bias_ih_l0.fill_(0.5)

    q_m = torch.ao.quantization.quantize_dynamic(m, {nn.LSTM}, dtype=torch.qint8)

    # Construct dict from float model
    wb_dict = {
        "weight_ih_l0": m.lstm.weight_ih_l0,
        "weight_hh_l0": m.lstm.weight_hh_l0,
        "bias_ih_l0": m.lstm.bias_ih_l0,
        "bias_hh_l0": m.lstm.bias_hh_l0,
    }

    try:
        q_m.lstm.set_weight_bias(wb_dict)
        print("Success! set_weight_bias accepted the dict.")
    except Exception as e:
        print(f"Failed with dict: {e}")


if __name__ == "__main__":
    test_lstm_dict()
