from utils import check_shape, CachedModule, PytorchTimer
from mem_linear import CachedLinear
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


with torch.autograd.profiler.profile(use_cuda=True) as prof:
    with torch.no_grad():
        A = torch.randn(4, 5).cuda()
        B = torch.randn(5, 6).cuda()
        C = A @ B
print(prof.key_averages().table(sort_by="cuda_time_total"))
prof.export_chrome_trace("test_GPU.json")