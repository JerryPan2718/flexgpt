from utils import check_shape, CachedModule, PytorchTimer
from mem_selfattn import CachedSelfAttn
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging


class MemBlock(nn.Module):
    """ a MemTransformer block """
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.H)
        self.ln2 = nn.LayerNorm(config.H)
        self.attn = CachedSelfAttn(config.K, config.H)
        self.mlp = nn.Sequential(
            nn.Linear(config.H, 4 * config.H),
            nn.GELU(),
            nn.Linear(4 * config.H, config.H),
            nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x):
        print(f"x:{type(x)}")
        print(f"x:{type(self.attn(self.ln1(x))[0])}")
        x = x + self.attn(self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x


# if __name__ == "__main__":
#     embd_pdrop = 0.1
#     resid_pdrop = 0.1
#     attn_pdrop = 0.1

#     B, K, T, H = (16, 12, 512, 768)
#     n_gen = T
#     layer = CachedSelfAttn(K, H).cuda()
#     # x = drop(...)
#     drop = nn.Dropout(embd_pdrop)
#     blocks = nn.Sequential(*[MemBlock(K, H) for _ in range(12)]).cuda()
#     with PytorchTimer(verbose=True):
#         x = blocks(x)