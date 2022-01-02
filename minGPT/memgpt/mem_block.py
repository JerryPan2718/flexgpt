from utils import check_shape, CachedModule, PytorchTimer, check_device_on_cuda
from mem_selfattn import CachedSelfAttn
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging


class MemBlock(nn.Module):
    """ a MemTransformer block """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln1 = nn.LayerNorm(config.H)
        self.ln2 = nn.LayerNorm(config.H)
        self.attn = CachedSelfAttn(config.K, config.H, cache_length=128)
        self.mlp = nn.Sequential(
            nn.Linear(config.H, 4 * config.H),
            nn.GELU(),
            nn.Linear(4 * config.H, config.H),
            nn.Dropout(config.resid_pdrop),
        )
        self.B_idx = 0

    def forward(self, x):
        if check_device_on_cuda(x) == False:
            print(f"mem_block.input.device: {x.device}")
        B, T, H = x.shape

        if self.B_idx == self.config.B:
            print(f"mem_block.B_idx: {self.B_idx}")
            self.attn.clear_cache()
            self.attn.reset_cache_counter()
            self.B_idx = 0
        
        # print(f"before forward: {x.shape}")
        y = self.attn(self.ln1(x))[0]
        x = x + y
        x = x + self.mlp(self.ln2(x))
        y_new = torch.randn((B, 1, H), device=self.attn.device)
        x = check_shape(torch.cat((y, y_new), dim=-2), (B, T + 1, H))

        self.B_idx += 1 
        # print(f"after forward: {x.shape}")

        if check_device_on_cuda(x) == False:
            print(f"mem_block.output.device: {x.device}")
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