import torch
import torch.nn as nn
from liger_kernel.ops.rms_norm import LigerRMSNormFunction
import os

def use_liger(cls):
    def wrapper(*args, **kwargs):
        if os.getenv("LIGER") == "1":
            print("Using LigerRMSNorm")
            return LigerRMSNorm(*args, **kwargs)
        else:
            print("Using PyTorch RMSNorm")
            return cls(*args, **kwargs)
    return wrapper


class LigerRMSNorm(nn.Module):
    def __init__(
        self, hidden_size, eps=1e-6, offset=0.0, casting_mode="llama", init_fn="ones"
    ):
        super().__init__()
        assert init_fn in [
            "ones",
            "zeros",
        ], f"init_fn must be either 'ones' or 'zeros', got {init_fn}"
        self.weight = nn.Parameter(
            torch.ones(hidden_size) if init_fn == "ones" else torch.zeros(hidden_size)
        )
        self.variance_epsilon, self.offset, self.casting_mode = (
            eps,
            offset,
            casting_mode,
        )

    def forward(self, hidden_states):
        return LigerRMSNormFunction.apply(
            hidden_states,
            self.weight,
            self.variance_epsilon,
            self.offset,
            self.casting_mode,
        )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}, offset={self.offset}"

@use_liger
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight