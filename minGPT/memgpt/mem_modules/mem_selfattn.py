from utils import check_shape, CachedModule, PytorchTimer
from mem_linear import CachedLinear
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import torch.cuda.profiler as profiler
import logging
logging.basicConfig(level=logging.DEBUG)


class CachedSelfAttn(CachedModule):
    def __init__(self, n_head, n_hidden, dropout=0.1, max_t=2048):
        """
        q: BKT(H/K)
        k: BKT(H/K)
        v: BKT(H/K)
        qkt: BKTT
        """
        super().__init__(dict(qkt=None, y=None))
        assert n_hidden % n_head == 0, "linear layer dimension is not divisible by n_head"
        self.q = CachedLinear(n_hidden, n_hidden)
        self.k = CachedLinear(n_hidden, n_hidden)
        self.v = CachedLinear(n_hidden, n_hidden)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.proj = CachedLinear(n_hidden, n_hidden)
        self.register_buffer("mask", torch.tril(torch.ones(max_t, max_t)).view(1, 1, max_t, max_t))
        self.n_head = n_head
        self.n_hidden = n_hidden
    
    def clear_cache(self):
        self.q.clear_cache()
        self.k.clear_cache()
        self.v.clear_cache()
        self.proj.clear_cache()
        self.cache = {}
    
    def set_cache(self, key, value):
        self.cache[key] = value
    
    def get_cache(self, key, device=None):
        val = self.cache.get(key, None) if self.cache else None
        if val is not None and device is not None:
            val = val.to(device)
        return val
        
    def forward_uncached(self, x):
        B, T, H = x.size()
        K = self.n_head
        
        q = self.q(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)

        qkt = q @ k.transpose(-2, -1) 
        att = qkt * (1.0 / math.sqrt(k.size(-1)))
        # attn = attn.to(x.device)
        mask = self.mask[:, :, :T, :T].to(x.device)
        att = att.masked_fill(mask == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        return qkt, y
    
    def forward_cached(self, x, qkt_cached, y_cached):
        B, T, H = x.size()
        K = self.n_head
        qkt_cached = check_shape(qkt_cached, (B, K, T - 1, T - 1))
        y_cached = check_shape(y_cached, (B, K, T - 1, H // K))
        
        q = self.q(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_head, H // self.n_head).transpose(1, 2)

        qkt = torch.zeros(B, K, T, T).to(x.device)
        qkt[:, :, :-1, :-1] = qkt_cached

        # qkt: BKT(H/K) * BKT(H/K).T -> BKTT
        qkt[:, :, -1:, :] = q[:, :, -1:, :] @ k.transpose(-2, -1)
        attn = qkt * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.to(x.device)
        mask = self.mask[:, :, :T, :T].to(x.device)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        new_attn = attn[:, :, -1:, :]

        # y_new: BK1T * BKT(H/K) -> BK1(H/K)
        y_new = new_attn @ v
        # y: stack(BK1(H/K), BK(T-1)(H/K)) -> BKT(H/K)
        y = torch.cat((y_cached, y_new), dim=-2)

        return qkt, y

    def forward(self, x):
        B, T, H = x.size()
        K = self.n_head
        assert H == self.n_hidden
        assert H % K == 0

        qkt_cached = self.get_cache("qkt", device=x.device)
        y_cached = self.get_cache("y", device=x.device)

        if y_cached is None or qkt_cached is None:
            self.clear_cache()
            qkt, y = self.forward_uncached(x)
            self.set_cache("qkt", check_shape(qkt, (B, K, T, T)))
            self.set_cache("y", check_shape(y, (B, K, T, H // K)))
        else:
            qkt, y = self.forward_cached(x, qkt_cached, y_cached)
            self.set_cache("qkt", check_shape(qkt, (B, K, T, T)))
            self.set_cache("y", check_shape(y, (B, K, T, H // K)))

        y = y.transpose(1, 2).contiguous().view(B, T, H)
        return y


if __name__ == "__main__":
    B, K, T, H = (16, 12, 128, 768)
    n_gen = T
    layer = CachedSelfAttn(K, H).cuda()
    x = torch.randn((B, T, H)).cuda()

    def bench(module, x, n_gen):
        x = x.cuda()
        B, T, H = x.shape
        module.clear_cache()
        with torch.inference_mode():
            with PytorchTimer(verbose=False) as t:
                for i in range(1, n_gen + 1):
                    y = check_shape(layer(x.cuda()), x.shape)
                    y_new = torch.randn((B, 1, H)).cuda()
                    x = check_shape(torch.cat((y, y_new), dim=-2).cuda(), (B, T + i, H))
        return t.elapsed
    
    def bench_chrome_trace(module, x, n_gen):
        x = x.cuda()
        B, T, H = x.shape
        module.clear_cache()
        with torch.inference_mode():
            with torch.autograd.profiler.profile() as prof:
                with torch.no_grad():
                    for i in range(1, n_gen + 1):
                        y = check_shape(layer(x.cuda()), x.shape)
                        y_new = torch.randn((B, 1, H)).cuda()
                        x = check_shape(torch.cat((y, y_new), dim=-2).cuda(), (B, T + i, H))
            print(prof.key_averages().table(sort_by="cpu_time_total"))
            prof.export_chrome_trace("bench_cached.json")
        return 

    def bench_uncached(module, x, n_gen):
        x = x.cuda()
        B, T, H = x.shape
        module.clear_cache()
        with torch.inference_mode():
            with PytorchTimer(verbose=False) as t:
                for i in range(1, n_gen + 1):
                    module.clear_cache()
                    y = check_shape(layer(x.cuda()), x.shape)
                    y_new = torch.randn((B, 1, H)).cuda()
                    x = check_shape(torch.cat((y, y_new), dim=-2).cuda(), (B, T + i, H))
        return t.elapsed

    def bench_uncached_chrome_trace(module, x, n_gen, do_profile=False):
        x = x.cuda()
        B, T, H = x.shape
        module.clear_cache()

        if do_profile:
            n_gen = 1

        with torch.inference_mode():
            with torch.autograd.profiler.profile() as prof:
                with torch.no_grad():
                    for i in range(1, n_gen + 1):
                        module.clear_cache()
                        y = check_shape(layer(x.cuda()), x.shape)
                        y_new = torch.randn((B, 1, H)).cuda()
                        x = check_shape(torch.cat((y, y_new), dim=-2).cuda(), (B, T + i, H))
            print(prof.key_averages().table(sort_by="cpu_time_total"))
            prof.export_chrome_trace("bench_uncached.json")
        return 

    # warmup
    for i in tqdm(range(4)):
        bench_uncached(layer, x, n_gen)
        
    # bench
    iters = []
    for i in tqdm(range(8)):
        iters.append(bench_uncached(layer, x, n_gen))
    
    
    mean = np.mean(iters)
    stddev = np.std(iters)
    print(f"Runtime w/o cache: {mean:.2f} +- {stddev:.2f}ms")

    bench_uncached_chrome_trace(layer, x, 2)

    # warmup
    for i in tqdm(range(4)):
        bench(layer, x, n_gen)

    # bench
    iters = []
    for i in tqdm(range(8)):
        iters.append(bench(layer, x, n_gen))

    mean = np.mean(iters)
    stddev = np.std(iters)
    print(f"Runtime w/ cache: {mean:.2f} +- {stddev:.2f}ms")

    bench_chrome_trace(layer, x, 2)