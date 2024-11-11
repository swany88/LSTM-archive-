import torch

# Check if CUDA is available
print(torch.cuda.is_available())

# Get the number of available GPUs
print(torch.cuda.device_count())

# Get the name of the current GPU
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
