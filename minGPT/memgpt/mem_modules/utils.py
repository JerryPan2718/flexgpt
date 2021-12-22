import torch
import torch.nn as nn
import logging
logging.basicConfig(level=logging.DEBUG)


def check_shape(x, shape):
    if x.shape != shape:
        logging.critical(f"Tensor shape {x.shape} does not match expected {shape}")
        assert False
    return x

class FLOPsCounters(object):
    def __enter__(self):
        self.FLOPs = 0

    def __add__(self):
        self.FLOPs += 1
    
    def __multplipy__(self):
        self.FLOPs += 1
    
    def __exit__(self):
        print(self.FLOPs)

class PytorchTimer(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        
    def __enter__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.start.record()
        return self
        
    def __exit__(self, *args):
        self.end.record()
        torch.cuda.synchronize()
        self.elapsed_secs = self.start.elapsed_time(self.end)
        self.elapsed = self.elapsed_secs  # millisecs
        if self.verbose:
            print('elapsed time: %f ms' % self.elapsed)


class CachedModule(nn.Module):
    """ cache object """
    def __init__(self, x=None):
        super().__init__()
        self.cache = x
    
    def clear_cache(self):
        self.cache = None
    
    def set_cache(self, x):
        self.cache = x
        
