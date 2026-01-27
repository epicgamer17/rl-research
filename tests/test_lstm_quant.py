import torch
import torch.nn as nn
import torch.ao.quantization


class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(10, 20, num_layers=1)


def test_lstm_update():
    torch.backends.quantized.engine = "qnnpack"
    # 1. Float Model
    float_model = LSTMModel()

    # 2. Quantize
    q_model = torch.ao.quantization.quantize_dynamic(
        float_model, {nn.LSTM}, dtype=torch.qint8
    )

    # 3. Check type
    print(f"Quantized LSTM type: {type(q_model.lstm)}")
    print(f"Attributes: {dir(q_model.lstm)}")

    # 4. Try to access packed params
    if hasattr(q_model.lstm, "_packed_params"):
        print("Has _packed_params")
        pp = q_model.lstm._packed_params
        print(f"Packed params type: {type(pp)}")
        print(f"Packed params attributes: {dir(pp)}")

        # Check if set_weights exists
        if hasattr(pp, "set_weights"):
            print("Found set_weights!")

    else:
        print("No _packed_params found")


if __name__ == "__main__":
    test_lstm_update()
