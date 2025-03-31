import torch.nn as nn
from liger_modules import RMSNorm

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.scale = dim**-0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm = RMSNorm(dim, eps=1e-6)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(B, N, self.heads, -1).transpose(1, 2), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.norm(out)