import torch
import sys
import os

# Add the project root to sys.path to import losses
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from basic_losses import categorical_crossentropy, kl_divergence

def test_assertions():
    # Example values from the user request that caused the error
    # Sum = 1.0018999576568604
    predicted = torch.tensor([[0.5039, 0.4980]], dtype=torch.float32) 
    target = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    print(f"Predicted sum (float32): {torch.sum(predicted, dim=-1).item()}")

    # 1. Test that it fails with float32 (current behavior)
    try:
        categorical_crossentropy(predicted, target)
        print("FAILED: categorical_crossentropy should have raised AssertionError with float32")
    except AssertionError as e:
        print(f"SUCCESS: Caught expected AssertionError with float32: {e}")

    # 2. Test with float16 (this should PASS after the fix)
    print("\nTesting with float16...")
    predicted_f16 = predicted.to(torch.float16)
    target_f16 = target.to(torch.float16)
    try:
        categorical_crossentropy(predicted_f16, target_f16)
        print("SUCCESS: categorical_crossentropy passed with float16")
    except AssertionError as e:
        print(f"FAILED: categorical_crossentropy raised AssertionError with float16: {e}")

    # 3. Test with bfloat16 (this should PASS after the fix)
    print("\nTesting with bfloat16...")
    predicted_bf16 = predicted.to(torch.bfloat16)
    target_bf16 = target.to(torch.bfloat16)
    try:
        categorical_crossentropy(predicted_bf16, target_bf16)
        print("SUCCESS: categorical_crossentropy passed with bfloat16")
    except AssertionError as e:
        print(f"FAILED: categorical_crossentropy raised AssertionError with bfloat16: {e}")

if __name__ == "__main__":
    test_assertions()

if __name__ == "__main__":
    test_assertions()
