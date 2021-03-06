import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from utils import PytorchTimer
import torch.autograd.profiler as profiler


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()

    mem_usage = []
    runtime = []
    t_elapsed = []
    for k in range(steps):
        # crop context if needed
        # x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        x_cond = x if x.size(1) <= block_size else x[:, :] # Use all previous tokens as Auto-regressive model
        
        # print(f"sample x_cond.shape: {x_cond.shape}")
        with PytorchTimer(verbose=False) as t:
            with torch.autograd.profiler.record_function("GPT Model Forward"):
                logits, _ = model(x_cond)
        runtime.append(t.elapsed)
        # print(f"mem_utils GPT forward pass: {t.elapsed}")
        with PytorchTimer(verbose=False) as t:
            # pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = top_k_logits(logits, top_k)
            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            if sample:
                ix = torch.multinomial(probs, num_samples=1)
                # print(ix.shape)
            else:
                _, ix = torch.topk(probs, k=1, dim=-1)
            # append to the sequence and continue
            x = torch.cat((x, ix), dim=1)
            # print(f"mem_utils: {x.shape}")
        print(f"mem_utils: {x_cond.shape}")
        # print(f"sample x.shape: {x.shape}")
        # print(f"mem_utils rest: {t.elapsed}")
        t_elapsed.append(t.elapsed)
        if len(t_elapsed) > 30:
            t_elapsed.pop(0)
        print(f"t_elapsed last 30 entries:{np.mean(t_elapsed)}")

        mem_usage.append(torch.cuda.memory_allocated())


    return x, (np.sum(mem_usage), np.sum(runtime))
