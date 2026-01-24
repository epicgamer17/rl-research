
import torch
import torch.nn as nn
import torch.ao.quantization
import inspect

def check_signature():
    torch.backends.quantized.engine = 'qnnpack'
    model = nn.LSTM(10, 20)
    q_model = torch.ao.quantization.quantize_dynamic(model, {nn.LSTM}, dtype=torch.qint8)
    
    print(inspect.signature(q_model.set_weight_bias))
    print(inspect.getdoc(q_model.set_weight_bias))

if __name__ == "__main__":
    check_signature()
