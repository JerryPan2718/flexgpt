import numpy as np
import math
import torch
import torch.profiler
import torch.nn as nn
from torch.nn import functional as F


with torch.profiler.profile(
    schedule=torch.profiler.schedule(
        wait=2,
        warmup=2,
        active=6,
        repeat=1),
    # on_trace_ready=trace_handler,
    on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name="./logs"),
    record_shapes=True,
    with_stack=True,
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
) as prof:
    # with torch.no_grad():
    A = torch.randn(4, 5).cuda()
    B = torch.randn(5, 6).cuda()
    C = A @ B
if prof is not None:
    # print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
else:
    print(f"Prof is None")
print(prof) 
# print(prof.key_averages().table(sort_by="cuda_time_total"))
# prof.export_chrome_trace("test_GPU.json")
