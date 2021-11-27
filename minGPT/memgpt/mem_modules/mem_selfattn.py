from utils import check_shape, CachedModule, PytorchTimer
from mem_linear import CachedLinear
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from loguru import logger


class CachedSelfAttn(CachedLinear):
    """" cached attention op """
    def __init__(self, config, q=None, k=None, v=None, qkt=None):
        """
        q: BKT(H/K)
        k: BKT(H/K)
        v: BKT(H/K)
        qkt: BKTT
        """
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.k = CachedLinear(config.n_embd, config.n_embd, False)
        self.q = CachedLinear(config.n_embd, config.n_embd, False)
        self.v = CachedLinear(config.n_embd, config.n_embd, False)
        self.y = CachedLinear(config.n_embd, config.n_embd, False)
        self.qkt = CachedLinear(config.n_embd, config.n_embd, False)
        self.n_head = config.n_head
        
    
    def clear_cache(self):
        self.q.clear_cache()
        self.k.clear_cache()
        self.v.clear_cache()
        self.y.clear_cache()
        self.qkt.clear_cache()
    
    def forward(self, x):
        B, T, H = x.size()
        K = self.n_head
        # print(T, K)
        # assert T % K == 0
        
        qkt_cached = self.qkt
        y_cached = self.y
        q = self.q(x).view(B, K, T, T // K)
        k = self.k(x).view(B, K, T, T // K)
        v = self.v(x).view(B, K, T, T // K)

        qkt = torch.zeros(B, K, T, T)
        qkt[:, :, :-1, :-1] = qkt_cached

        # qkt: BKT(H/K) * BKT(H/K).T -> BKTT
        qkt[:, :, :, -1] = q[:, :, :, -1:] @ k.transpose(-2, -1) 
        attn = qkt * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(self.attn_mask[:, :, :T, :T], 1e-9)
        attn = F.softmax(attn, dim=-1)
        new_attn = attn[:, :, -1:, -1:]

        # y_new: BK(T-1)T * BKT(H/K) -> BK(T-1)(H/K)
        y_new = new_attn @ v
        y = torch.stack(y_cached, y_new, dim=2)

        # Clear cache before set cache
        self.clear_cache()
        self.qkt.set_cache(qkt)
        self.y.set_cache(y)
        return y
        