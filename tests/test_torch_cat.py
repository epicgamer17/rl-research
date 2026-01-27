import torch

try:
    existing = torch.empty(0)
    print(f"Existing shape: {existing.shape}, ndim: {existing.ndim}")

    new_data = [0.1, 0.2, 0.3]
    to_append = torch.tensor([new_data])
    print(f"To append shape: {to_append.shape}, ndim: {to_append.ndim}")

    result = torch.cat((existing, to_append))
    print("Success!")
    print(f"Result shape: {result.shape}")
except Exception as e:
    print(f"Failed: {e}")

print("-" * 20)

try:
    # Simulating what we might need to do: Handle first append
    existing = torch.empty(0)
    to_append = torch.tensor([[0.1, 0.2, 0.3]])  # 1x3

    if existing.numel() == 0:
        result = to_append
    else:
        result = torch.cat((existing, to_append))

    print("Manual check success!")
    print(f"Result shape: {result.shape}")

    # Second append
    to_append2 = torch.tensor([[0.4, 0.5, 0.6]])
    result = torch.cat((result, to_append2))
    print(f"Result after 2nd append shape: {result.shape}")

except Exception as e:
    print(f"Manual check failed: {e}")
