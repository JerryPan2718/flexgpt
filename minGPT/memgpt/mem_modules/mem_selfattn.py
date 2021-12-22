from utils import check_shape, CachedModule, PytorchTimer
from mem_linear import CachedLinear
import time
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import torch.cuda.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
# from pypapi import papi_high as high
# from pypapi import events as papi_events


import logging
logging.basicConfig(level=logging.DEBUG)

CUDA_VISIBLE_DEVICES = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class CachedSelfAttn(CachedModule):
    def __init__(self, n_head, n_hidden, dropout=0.1, max_t=2048, cache_length=64):
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
        self.cache_counter = 0
        self.cache_length = cache_length
    
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
        
    def reset_cache_counter(self):
        self.cache_counter = 0
    
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

        qkt = torch.zeros(B, K, T, T, device=x.device)
        qkt[:, :, :-1, :-1] = qkt_cached

        # qkt: BK1(H/K) * BK(H/K)T -> BK1T
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

        if (y_cached is None or qkt_cached is None) or self.cache_counter >= self.cache_length:
            self.clear_cache()
            qkt, y = self.forward_uncached(x)
            self.set_cache("qkt", check_shape(qkt, (B, K, T, T)))
            self.set_cache("y", check_shape(y, (B, K, T, H // K)))
        else:
            qkt, y = self.forward_cached(x, qkt_cached, y_cached)
            self.set_cache("qkt", check_shape(qkt, (B, K, T, T)))
            self.set_cache("y", check_shape(y, (B, K, T, H // K)))

        y = y.transpose(1, 2).contiguous().view(B, T, H)
        self.cache_counter += 1
        return y

### Helper function for Benchmark ###
def bench_cached(module, x, n_gen):
    x = x.to(device)
    B, T, H = x.shape
    mem_usage = []
    module.clear_cache()
    module.reset_cache_counter()
    with torch.inference_mode():
        with PytorchTimer(verbose=False) as t:
            for i in range(1, n_gen + 1):
                y = check_shape(module(x.to(device)), x.shape)
                y_new = torch.randn((B, 1, H)).to(device)
                x = check_shape(torch.cat((y, y_new), dim=-2).to(device), (B, T + i, H))
                mem_usage.append(torch.cuda.memory_allocated())
    return t.elapsed, mem_usage

def bench_uncached(module, x, n_gen):
    x = x.to(device)
    B, T, H = x.shape
    mem_usage = []
    module.clear_cache()
    module.reset_cache_counter()
    with torch.inference_mode():
        with PytorchTimer(verbose=False) as t:
            for i in range(1, n_gen + 1):
                module.clear_cache()
                y = check_shape(module(x.to(device)), x.shape)
                y_new = torch.randn((B, 1, H)).to(device)
                x = check_shape(torch.cat((y, y_new), dim=-2).to(device), (B, T + i, H))
                mem_usage.append(torch.cuda.memory_allocated())
    return t.elapsed, mem_usage

def pipeline(benchmark_function, module):
        # warmup
        for i in tqdm(range(4)):
            benchmark_function(module, x, Tg)

        # bench
        total_time = []
        mem_usage = []
        # FLOP = []
        for i in tqdm(range(8)):
            # if i == 7:
            #     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            #         ret = benchmark_function(module, x, Tg)
            #         total_time.append(ret[0])    
            #         mem_usage += ret[1]
            #         # FLOP = ret[2]
            #     # prof.export_chrome_trace(f"trace-token_length={x.size(1)} mem_length={module.cache_length}.json")

            # else:
            ret = benchmark_function(module, x, Tg)
            total_time.append(ret[0])    
            mem_usage += ret[1]
        
        
        return [np.mean(total_time), np.std(total_time), np.mean(mem_usage) / 10 ** 6, np.std(mem_usage) / 10 ** 6]


if __name__ == "__main__":
    d = {}
    Tcs = [1024, 512, 256, 128] # 128, 256, 512, 1024
    B, K, _, H = (2, 4, 128, 128)
    for Tc in Tcs:
        Tg = Tc 
        layer0 = CachedSelfAttn(K, H, cache_length=0).to(device)
        layer1 = CachedSelfAttn(K, H, cache_length=0.25 * Tg).to(device)
        layer2 = CachedSelfAttn(K, H, cache_length=0.5 * Tg).to(device)
        layer3 = CachedSelfAttn(K, H, cache_length=0.75 * Tg).to(device)
        layer4 = CachedSelfAttn(K, H, cache_length=Tg).to(device)

        x = torch.randn((B, Tc, H)).to(device)
        ret0 = pipeline(bench_cached, layer0)
        d[f"Tc={Tc} Tg={Tg} cache_length={0}"] = ret0
        torch.cuda.empty_cache()

        x = torch.randn((B, Tc, H)).to(device)
        ret1 = pipeline(bench_cached, layer1)
        d[f"Tc={Tc} Tg={Tg} cache_length={0.25 * Tg}"] = ret1
        torch.cuda.empty_cache()

        x = torch.randn((B, Tc, H)).to(device)
        ret2 = pipeline(bench_cached, layer2)
        d[f"Tc={Tc} Tg={Tg} cache_length={0.5 * Tg}"] = ret2
        torch.cuda.empty_cache()

        x = torch.randn((B, Tc, H)).to(device)
        ret3 = pipeline(bench_cached, layer3)
        d[f"Tc={Tc} Tg={Tg} cache_length={0.75 * Tg}"] = ret3
        torch.cuda.empty_cache()

        x = torch.randn((B, Tc, H)).to(device)
        ret4 = pipeline(bench_cached, layer4)
        d[f"Tc={Tc} Tg={Tg} cache_length={Tg}"] = ret4
        torch.cuda.empty_cache()

    print(d)
    df = pd.DataFrame(data=d, index=["runtime_mean(ms)", "runtime_std(ms)", "mem_mean(MB)", "mem_std(MB)"])
    print(df)
    df.to_csv("mem_selfattn.csv")

    ############################################################## Works
    # B, K, Tc, H = (16, 12, 128, 768)
    # Tg = Tc 

    # layer0 = CachedSelfAttn(K, H, cache_length=0).cuda()
    # layer1 = CachedSelfAttn(K, H, cache_length=32).cuda()
    # layer2 = CachedSelfAttn(K, H, cache_length=64).cuda()
    # layer3 = CachedSelfAttn(K, H, cache_length=96).cuda()
    # layer4 = CachedSelfAttn(K, H, cache_length=128).cuda()
    # x = torch.randn((B, Tc, H)).cuda()

    # # warmup
    # for i in tqdm(range(4)):
    #     bench_cached(layer0, x, Tg)

    # # bench
    # total_time = []
    # memUsage = []
    # for i in tqdm(range(8)):
    #     ret = bench_cached(layer0, x, Tg)
    #     total_time.append(ret[0])    
    #     memUsage += ret[1]


    # mean = np.mean(total_time)
    # stddev = np.std(total_time)
    # print(f"Runtime w/o cache: {mean:.2f} +- {stddev:.2f}ms")
    # print(f"memUsage w/o cache: {np.mean(memUsage) / 10 ** 6:.2f} +- {np.std(memUsage) / 10 ** 6:.2f}MB")

    # # bench_uncached_chrome_trace(layer, x, 2)

    # # warmup
    # for i in tqdm(range(4)):
    #     bench_cached(layer1, x, Tg)

    # # bench
    # total_time = []
    # memUsage = []
    # for i in tqdm(range(8)):
    #     ret = bench_cached(layer1, x, Tg)
    #     total_time.append(ret[0])    
    #     memUsage += ret[1]

    # mean = np.mean(total_time)
    # stddev = np.std(total_time)
    # print(f"Runtime w/ cache_length=32: {mean:.2f} +- {stddev:.2f}ms")
    # print(f"memUsage w/ cache_length=32: {np.mean(memUsage) / 10 ** 6:.2f} +- {np.std(memUsage) / 10 ** 6:.2f}MB")