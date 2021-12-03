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
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CachedSelfAttn(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            # nn.Dropout(config.resid_pdrop),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x



if __name__ == "__main__":
    B, K, T, H = (16, 12, 512, 768)
    n_gen = T
    layer = CachedSelfAttn(K, H).cuda()
    x = torch.randn((B, T, H)).cuda()