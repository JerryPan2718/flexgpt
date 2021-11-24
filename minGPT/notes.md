# cache object
```python
class CachedModule(nn.Module):
    def __init__(self):
        self.cache = None
    
    def clear_cache(self):
        self.cache = None
    
    def set_cache(self, x):
        self.cache = x
```

# cached nn.Dense layer
```python
class CachedDense(nn.Dense):
    def forward(self, x: BTH):
        cache: B(T-1)H = self.cache
        new_out = self.dense(x[:, -1:, :])
        y = torch.stack(cache, new_out)
        self.set_cache(y)
        return y
```

# cached attention op
```python
class CachedSelfAttn:
    def __init__():
        self.q = CachedDense
        self.k = CachedDense
        self.v = CachedDense
    
    def clear_cache(self):
        self.q.clear_cache()
        self.k.clear_cache()
        self.v.clear_cache()
    
    def forward(x):
        qkt_cached = self.cache[0]
        y_cached = self.cache[1]
        q, k, v = self.q(x), ...
        qkt = torch.zeros(B, K, T, T)
        qkt[:, :, :-1, :-1] = qkt_cached
        qkt[:, :, :, -1] = q[:, :, :, -1:] @ k.T
        qkt = qkt.masked_fill(self.attn_mask, 1e-9)
        attn = softmax(qkt)
        new_attn = attn[:, :, -1:, -1:]
        y = torch.stack(y_cached, new_attn @ v)
        self.set_cache((qkt, y))
        return y
```