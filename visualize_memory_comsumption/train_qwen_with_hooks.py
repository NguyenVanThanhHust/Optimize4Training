import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")

activation_sizes = []

def forward_hook(module, input, output):
    """
    Hook to calculate activation size for each module.
    """
    if isinstance(output, torch.Tensor):
        activation_sizes.append(output.numel() * output.element_size())
    elif isinstance(output, (tuple, list)):
        for tensor in output:
            if isinstance(tensor, torch.Tensor):
                activation_sizes.append(tensor.numel() * tensor.element_size())

# Register hooks for each submodule
hooks = []
for submodule in model.modules():
    hooks.append(submodule.register_forward_hook(forward_hook))

# Perform a forward pass with a dummy input
dummy_input = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
model.eval()  # No gradients needed for memory measurement
with torch.no_grad():
    model(dummy_input)

# Clean up hooks
for hook in hooks:
    hook.remove()

print(sum(activation_sizes))  # Output: 5065216