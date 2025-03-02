import torch

rand_tensor = torch.rand(4, 3, 28, 28, dtype=torch.float32)

print(rand_tensor.shape)
print(rand_tensor.view(rand_tensor.size(0), -1).shape)

def get_patch(x, patch_shape):
    c, (h, w) = x.shape[1], patch_shape
    return x.unfold(2,h,1).unfold(3, w, 1).transpose(1, 3).reshape(-1, c, h, w)

y = get_patch(rand_tensor, (7, 7))
print(y.shape)