import torch
from torch import nn

# Start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = nn.Linear(10_000, 50_000, device="cuda")
for _ in range(3):
    inputs = torch.randn(5_000, 10_000, device="cuda")
    outputs = model(inputs)

# Dump memory snapp shot history to a file and stop recording
torch.cuda.memory._dump_snapshot("profile.pkl")
torch.cuda.memory._record_memory_history(enabled=None)