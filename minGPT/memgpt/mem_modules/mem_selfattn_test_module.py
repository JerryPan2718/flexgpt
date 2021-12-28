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
import datetime
from mem_gpt_flops import selfattn_flop


today = datetime.date.today()
# from pypapi import papi_high as high
# from pypapi import events as papi_events


import logging
logging.basicConfig(level=logging.DEBUG)

CUDA_VISIBLE_DEVICES = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class CachedSelfAttn(CachedModule):
    def __init__(self, n_head, n_hidden, dropout=0.1, max_t=2048, cache_length=64, B=12, T=2048):
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
        # self.zero_pad = torch.zeros(B, K, T, T, device=device)
    
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
        
        with PytorchTimer(verbose=False) as T1:
            q = self.q(x).view(B, T, K, H // K).transpose(1, 2)
            k = self.k(x).view(B, T, K, H // K).transpose(1, 2)
            v = self.v(x).view(B, T, K, H // K).transpose(1, 2)
        t1 = T1.elapsed

        with PytorchTimer(verbose=False) as T2:
            qkt = q @ k.transpose(-2, -1) 
            att = qkt * (1.0 / math.sqrt(k.size(-1)))
        t2 = T2.elapsed
        # attn = attn.to(x.device)
        with PytorchTimer(verbose=False) as T3:
            mask = self.mask[:, :, :T, :T].to(x.device)
            att = att.masked_fill(mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        t3 = T3.elapsed
        return qkt, y, t1, t2, t3
    
    def forward_cached(self, x, qkt_cached, y_cached):
        B, T, H = x.size()
        K = self.n_head

        qkt_cached = check_shape(qkt_cached, (B, K, T - 1, T - 1))
        y_cached = check_shape(y_cached, (B, K, T - 1, H // K))

        with PytorchTimer(verbose=False) as T1:
            q = self.q(x).view(B, T, K, H // K).transpose(1, 2)
            k = self.k(x).view(B, T, K, H // K).transpose(1, 2)
            v = self.v(x).view(B, T, K, H // K).transpose(1, 2)

            # qkt = self.zero_pad[:B, :K, :T, :T]
            qkt = torch.zeros(B, K, T, T, device=x.device)
            qkt[:, :, :T-1, :T-1] = qkt_cached
        t1 = T1.elapsed

        # qkt: BK1(H/K) * BK(H/K)T -> BK1T
        with PytorchTimer(verbose=False) as T2:
            qkt[:, :, -1:, :] = q[:, :, -1:, :] @ k.transpose(-2, -1)
            attn = qkt * (1.0 / math.sqrt(k.size(-1)))
            attn = attn.to(x.device)
        t2 = T2.elapsed
        
        mask = self.mask[:, :, :T, :T].to(x.device)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        new_attn = attn[:, :, -1:, :]
        with PytorchTimer(verbose=False) as T3:
            # y_new: BK1T * BKT(H/K) -> BK1(H/K)
            y_new = new_attn @ v
            # y: stack(BK1(H/K), BK(T-1)(H/K)) -> BKT(H/K)
            y = torch.cat((y_cached, y_new), dim=-2)
        t3 = T3.elapsed
        
        return qkt, y, t1, t2, t3

    def forward(self, x):
        B, T, H = x.size()
        K = self.n_head
        assert H == self.n_hidden
        assert H % K == 0

        qkt_cached = self.get_cache("qkt", device=x.device)
        y_cached = self.get_cache("y", device=x.device)

        if (y_cached is None or qkt_cached is None) or self.cache_counter >= self.cache_length:
            self.clear_cache()
            qkt, y, t1, t2, t3 = self.forward_uncached(x)
            self.set_cache("qkt", check_shape(qkt, (B, K, T, T)))
            self.set_cache("y", check_shape(y, (B, K, T, H // K)))
        else:
            qkt, y, t1, t2, t3 = self.forward_cached(x, qkt_cached, y_cached)
            self.set_cache("qkt", check_shape(qkt, (B, K, T, T)))
            self.set_cache("y", check_shape(y, (B, K, T, H // K)))

        y = y.transpose(1, 2).contiguous().view(B, T, H)
        self.cache_counter += 1
        # print(t1, t2, t3)
        return y, t1, t2, t3

### Helper function for Benchmark ###
def bench_cached(module, x, n_gen):
    t1_array = []
    t2_array = []
    t3_array = []
    x = x.to(device)
    B, T, H = x.shape
    mem_usage = []
    module.clear_cache()
    module.reset_cache_counter()
    with torch.inference_mode():
        with PytorchTimer(verbose=False) as t:
            for i in range(1, n_gen + 1):
                y, t1, t2, t3 = module(x.to(device))
                y = check_shape(y, x.shape)
                t1_array.append(t1)
                t2_array.append(t2)
                t3_array.append(t3)
                y_new = torch.randn((B, 1, H)).to(device)
                x = check_shape(torch.cat((y, y_new), dim=-2).to(device), (B, T + i, H))
                mem_usage.append(torch.cuda.memory_allocated())
    print(np.sum(t1_array), np.sum(t2_array), np.sum(t3_array))
    return t.elapsed, mem_usage, np.sum(t1_array), np.sum(t2_array), np.sum(t3_array)

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
        t1_array = []
        t2_array = []
        t3_array = []
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
            t1_array.append(ret[2])
            t2_array.append(ret[3])
            t3_array.append(ret[4])
        
        return [np.mean(total_time), np.std(total_time), np.mean(mem_usage) / 10 ** 6, np.std(mem_usage) / 10 ** 6, np.mean(t1_array), np.std(t1_array), np.mean(t2_array), np.std(t2_array), np.mean(t3_array), np.std(t3_array)]


if __name__ == "__main__":
    # Parameters: layers, d_model
    # 117M: 12, 768
    # 345M: 24, 1024
    # 762M: 36, 1280
    # 1542M: 48, 1600
    hparams = {"117M": (12, 768), "345M": (24, 1024), "762M": (36, 1280), "1542M": (48, 1600)}
    start = time.time()
    for model_size, hparam in hparams.items():
        with torch.no_grad():
            with torch.autocast(device):
                d = {}
                Tcs = [512, 256, 128] # 1024, 512, 256, 128
                K = 4
                B, H = hparam
                for Tc in Tcs:
                    Tg = Tc 
                    layer0 = CachedSelfAttn(K, H, cache_length=0, B=B, T=Tc+Tg).to(device)
                    layer1 = CachedSelfAttn(K, H, cache_length=0.25 * Tg, B=B, T=Tc+Tg).to(device)
                    layer2 = CachedSelfAttn(K, H, cache_length=0.5 * Tg, B=B, T=Tc+Tg).to(device)
                    layer3 = CachedSelfAttn(K, H, cache_length=0.75 * Tg, B=B, T=Tc+Tg).to(device)
                    layer4 = CachedSelfAttn(K, H, cache_length=Tg, B=B, T=Tc+Tg).to(device)

                    x = torch.randn((B, Tc, H)).to(device)
                    ret0 = pipeline(bench_cached, layer0)
                    flops = selfattn_flop(B=B, H=H, K=K, Tc=Tc, Tg=Tg, cache_length=0)
                    print(ret0 + [flops])
                    d[f"Tc={Tc} Tg={Tg} cache_length={0}"] = ret0 + [flops]
                    torch.cuda.empty_cache()

                    x = torch.randn((B, Tc, H)).to(device)
                    ret1 = pipeline(bench_cached, layer1)
                    flops = selfattn_flop(B=B, H=H, K=K, Tc=Tc, Tg=Tg, cache_length=0.25 * Tg)
                    print(ret1 + [flops])
                    d[f"Tc={Tc} Tg={Tg} cache_length={0.25 * Tg}"] = ret1 + [flops]
                    torch.cuda.empty_cache()

                    x = torch.randn((B, Tc, H)).to(device)
                    ret2 = pipeline(bench_cached, layer2)
                    flops = selfattn_flop(B=B, H=H, K=K, Tc=Tc, Tg=Tg, cache_length=0.5 * Tg)
                    print(ret2 + [flops])
                    d[f"Tc={Tc} Tg={Tg} cache_length={0.5 * Tg}"] = ret2 + [flops]
                    torch.cuda.empty_cache()

                    x = torch.randn((B, Tc, H)).to(device)
                    ret3 = pipeline(bench_cached, layer3)
                    flops = selfattn_flop(B=B, H=H, K=K, Tc=Tc, Tg=Tg, cache_length=0.75 * Tg)
                    print(ret3 + [flops])
                    d[f"Tc={Tc} Tg={Tg} cache_length={0.75 * Tg}"] = ret3 + [flops]
                    torch.cuda.empty_cache()

                    x = torch.randn((B, Tc, H)).to(device)
                    ret4 = pipeline(bench_cached, layer4)
                    flops = selfattn_flop(B=B, H=H, K=K, Tc=Tc, Tg=Tg, cache_length=Tg)
                    print(ret4 + [flops])
                    d[f"Tc={Tc} Tg={Tg} cache_length={Tg}"] = ret4 + [flops]
                    torch.cuda.empty_cache()

        print(d)
        df = pd.DataFrame(data=d, index=["runtime_mean(ms)", "runtime_std(ms)", "mem_mean(MB)", "mem_std(MB)", "t1_mean(s)", "t1_std(s)", "t2_mean(s)", "t2_std(s)","t3_mean(s)", "t3_std(s)", "flops"])
        print(df)
        df.to_csv(f"logs/{today}-mem_selfattn_{model_size}_K={K}_test_nograd_AMP.csv")
    print(time.time() - start)