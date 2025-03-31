import torch
from modules import Attention

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = torch.randn(2,10,64, device = device)
attn = Attention(dim=64, heads=8).to(device)
print(attn)

output = attn(x)
print("Input Shape: ", x.shape)
print("Output Shape: ", output.shape)

'''
Two ways to run:
    1. Default to PyTorch: python test.py 
    2. Use Liger Kernels: LIGER=1 python test.py
'''