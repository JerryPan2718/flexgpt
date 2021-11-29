from utils import check_shape, CachedModule, PytorchTimer
from mem_linear import CachedLinear
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from loguru import logger


class CachedSelfAttn(CachedLinear):
    """" cached self-attention """
    def __init__(self, n_head, in_features=768, out_features=768, bias=False, q=None, k=None, v=None):
        """
        q: BKT(H/K)
        k: BKT(H/K)
        v: BKT(H/K)
        qkt: BKTT
        """
        assert in_features % n_head == 0, "linear layer dimension is not divisible by n_head"

        CachedLinear.__init__(self, in_features, out_features, False)
        self.cache = {"q": q, "k": k, "v": v, "y": None, "qkt": None}
        # self.q = q
        # self.k = k
        # self.v = v
        # self.y = CachedModule(self)
        # self.qkt = CachedModule(self)
        self.register_buffer("mask", torch.tril(torch.ones(in_features, in_features)).view(1, 1, in_features, in_features))
        self.n_head = n_head
        
    def clear_cache(self):
        for key, val in self.cache.items():
            if val is not None:
                # print(val)
                # self.cache[key].clear_cache()
                # val.clear_cache()
                self.cache[key] = None
    
    def forward(self, x):
        B, T, H = x.size()
        K = self.n_head

        qkt_cached = self.cache["qkt"]
        y_cached = self.cache["y"]

        if y_cached is None:
            q = self.cache["q"](x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
            k = self.cache["k"](x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
            v = self.cache["v"](x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)

            print(q[:, :, :, -1:].shape, k.transpose(-2, -1).shape)
            qkt = q @ k.transpose(-2, -1) 
            att = qkt * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, H)

            self.cache["qkt"] = qkt
            self.cache["y"] = y
        else:
            q = self.cache["q"](x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
            k = self.cache["k"](x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
            v = self.cache["v"](x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)

            qkt = torch.zeros(B, K, T, T)
            qkt[:, :, :-1, :-1] = qkt_cached

            # qkt: BKT(H/K) * BKT(H/K).T -> BKTT
            print(q[:, :, :, -1:].shape, k.transpose(-2, -1).shape)
            qkt[:, :, :, -1:] = q[:, :, :, -1:] @ k.transpose(-2, -1)
            attn = qkt * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            new_attn = attn[:, :, -1:, -1:]

            # y_new: BK1T * BKT(H/K) -> BK1(H/K)
            y_new = new_attn @ v
            # y: stack(BK1(H/K), BK(T-1)(H/K)) -> BKT(H/K)
            y = torch.stack(y_cached, y_new, dim=2)

            # Clear cache before set cache
            self.clear_cache()
            self.cache["qkt"] = qkt
            self.cache["y"] = y

        return y


if __name__ == "__main__":
    B, T, H = (16, 128, 768)

    
    q = CachedLinear(H, H, False)
    k = CachedLinear(H, H, False)
    v = CachedLinear(H, H, False)

    layer = CachedSelfAttn(n_head=2, in_features=H, out_features=H, bias=False, q=q, k=k, v=v)
    x = torch.randn((B, T, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T, H))
    layer.clear_cache()

    layer = CachedSelfAttn(n_head=2, in_features=H, out_features=H, bias=False, q=q, k=k, v=v)
    x = torch.randn((B, T + 1, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T + 1, H))
    layer.clear_cache()

    layer = CachedSelfAttn(n_head=2, in_features=H, out_features=H, bias=False, q=q, k=k, v=v)
    x = torch.randn((B, T + 2, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T + 2, H))
    layer.clear_cache()

    logger.debug(f"test cache")
    layer = CachedSelfAttn(n_head=2, in_features=H, out_features=H, bias=False, q=q, k=k, v=v)
    x = torch.randn((B, T + 3, H))
    with PytorchTimer(verbose=True):
        y = layer(x) 
    