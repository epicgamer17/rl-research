
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import time
import copy
import sys

class FloatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)

def worker_fn(stop_flag, model):
    print("Worker started")
    # Ensure engine is set in worker too
    torch.backends.quantized.engine = 'qnnpack'
    while not stop_flag.value:
        # Access the weight (this is tricky for quantized models)
        # For quantized dynamic, we check packed params
        # But let's just run forward and see output
        with torch.no_grad():
            inp = torch.ones(1, 10)
            out = model(inp)
            print(f"Worker output sum: {out.sum().item()}")
        time.sleep(0.5)

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    # Set engine for main process
    try:
        torch.backends.quantized.engine = 'qnnpack'
    except Exception as e:
        print(f"Warning: Could not set qnnpack: {e}")

    # 1. Create Float Model
    float_model = FloatModel()
    with torch.no_grad():
        float_model.fc.weight.fill_(1.0)
        float_model.fc.bias.fill_(0.0)

    # 2. Quantize
    quantized_model = torch.ao.quantization.quantize_dynamic(
        float_model, {nn.Linear}, dtype=torch.qint8
    )
    quantized_model.eval()
    for p in quantized_model.parameters():
        p.requires_grad = False
    
    # 3. Share Memory
    # quantize_dynamic returns a new model. Does it support share_memory?
    quantized_model.share_memory()
    
    # 4. Start Worker
    stop_flag = mp.Value('i', 0)
    p = mp.Process(target=worker_fn, args=(stop_flag, quantized_model))
    p.start()
    
    time.sleep(1) # Let worker read 1.0s
    
    # 5. Update weights in main process
    print("Main: Updating weights...")
    new_float = FloatModel()
    with torch.no_grad():
        new_float.fc.weight.fill_(2.0)
        new_float.fc.bias.fill_(0.0)
        
    # HOW TO UPDATE in place?
    # We find the module and update packed params?
    # quantized_model.fc is a torch.nn.quantized.dynamic.modules.linear.Linear
    # It has _packed_params
    
    # We can try to repack from float
    # The set_weight_bias method might exist on the packed_params object or module
    
    q_linear = quantized_model.fc
    # q_linear.set_weight_bias(new_float.fc.weight, new_float.fc.bias) # This usually exists for static, check dynamic
    
    try:
        # Dynamic linear often exposes set_weight_bias
        q_linear.set_weight_bias(new_float.fc.weight, new_float.fc.bias)
        print("Updated weights using set_weight_bias")
    except Exception as e:
        print(f"set_weight_bias failed: {e}")
        # Try finding another way
        # q_linear._packed_params.set_weight_bias?
    
    time.sleep(2)
    stop_flag.value = 1
    p.join()
