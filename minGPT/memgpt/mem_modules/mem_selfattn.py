from utils import check_shape, CachedModule, PytorchTimer
from mem_linear import CachedLinear
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from loguru import logger


class CachedSelfAttn(CachedLinear):
    """" cached attention op """
    def __init__(self, n_head, in_features=768, out_features=768, bias=False, q=None, k=None, v=None):
        """
        q: BKT(H/K)
        k: BKT(H/K)
        v: BKT(H/K)
        qkt: BKTT
        """
        assert in_features % n_head == 0, "linear layer dimension is not divisible by n_head"

        CachedLinear.__init__(self, in_features, out_features, False)
        self.q = q
        self.k = k
        self.v = v
        self.y = CachedModule(self)
        self.qkt = CachedModule(self)
        self.register_buffer("mask", torch.tril(torch.ones(in_features, in_features))
                                     .view(1, 1, in_features, in_features))
        self.n_head = n_head
        
    
    def clear_cache(self):
        if self.q:
            self.q.clear_cache()
        if self.k:
            self.k.clear_cache()
        if self.v:
            self.v.clear_cache()
        if self.y:
            self.y.clear_cache()
        if self.qkt:
            self.qkt.clear_cache()
    
    def forward(self, x):
        B, T, H = x.size()
        K = self.n_head

        qkt_cached = self.qkt
        y_cached = self.y

        if y_cached is None:
            q = self.q(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
            k = self.k(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
            v = self.v(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)

            qkt = q @ k.transpose(-2, -1) 
            att = qkt * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, H)

            self.clear_cache()
            self.qkt.set_cache(qkt)
            self.y.set_cache(y)
        else:
            q = self.q(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
            k = self.k(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
            v = self.v(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)

            qkt = torch.zeros(B, K, T, T)
            qkt[:, :, :-1, :-1] = qkt_cached

            # qkt: BKT(H/K) * BKT(H/K).T -> BKTT
            qkt[:, :, :, -1] = q[:, :, :, -1:] @ k.transpose(-2, -1) 
            attn = qkt * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(self.attn_mask[:, :, :T, :T], 1e-9)
            attn = F.softmax(attn, dim=-1)
            new_attn = attn[:, :, -1:, -1:]

            # y_new: BK1T * BKT(H/K) -> BK1(H/K)
            y_new = new_attn @ v
            # y: stack(BK1(H/K), BK(T-1)(H/K)) -> BKT(H/K)
            y = torch.stack(y_cached, y_new, dim=2)

            # Clear cache before set cache
            self.clear_cache()
            self.qkt.set_cache(qkt)
            self.y.set_cache(y)

        return y


if __name__ == "__main__":
    B, T, H = (16, 128, 768)

    x = torch.randn((B, T, H))
    q = CachedLinear(H, H, False)
    k = CachedLinear(H, H, False)
    v = CachedLinear(H, H, False)
    layer = CachedSelfAttn(n_head=2, in_features=H, out_features=H, bias=False, q=q, k=k, v=v)
    layer.clear_cache()

    with PytorchTimer(verbose=True):
        y = layer(x)

    # with PytorchTimer(verbose=True):
    #     y = layer(x)