# set up logging
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
)

# make deterministic
from mem_utils import set_seed
from utils import check_shape, CachedModule, PytorchTimer, check_device_on_cuda
set_seed(2718)
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset
import time
import datetime
from torch.profiler import profile, record_function, ProfilerActivity
import pandas as pd

today = datetime.date.today()
CUDA_VISIBLE_DEVICES = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data
    
    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
block_size = 2048 # spatial extent of the model for its context
# you can download this file at https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt
text = open('input.txt', 'r').read() # don't worry we won't run out of file handles
train_dataset = CharDataset(text, block_size) # one line of poem is roughly 50 characters

def model_init(B, K, H, cache_length):
    from mem_gpt import MemGPT, MemGPTConfig
    mem_config = MemGPTConfig(train_dataset.vocab_size, train_dataset.block_size,
        B=12, K=4, H=768, cache_length=0)
    model = MemGPT(mem_config)
    print("=" * 50)

    from mem_trainer import MemTrainer, MemTrainerConfig

    # initialize a trainer instance and kick off training
    tconf = MemTrainerConfig(max_epochs=1, batch_size=128, learning_rate=6e-4,
                        lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*block_size,
                        num_workers=4)
    trainer = MemTrainer(model, train_dataset, None, tconf)
    trainer.train()
    print("=" * 50)
    return model, trainer

def model_sampling(model, trainer, steps):
    from mem_utils import sample
    context = "O God, O God!"
    x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(trainer.device)
    # print(f"Tc: {x.shape[1]}")
    y = sample(model, x, steps, temperature=1.0, sample=True, top_k=10)[0]
    return y

if __name__ == "__main__":
    hparams = {"117M": (12, 768), "345M": (24, 1024), "762M": (36, 1280), "1542M": (48, 1600)}
    cache_lengths = [0, 0.25, 0.5, 0.5, 1]
    d = {}
    Tgs = [256, 512, 1024]
    start = time.time()
    for model_size, hparam in hparams.items():
        B, H = hparam
        K = 4
        for Tg in Tgs:
            for cache_length in cache_lengths:
                with torch.no_grad():
                    with torch.autocast(device):
                        model, trainer = model_init(B, K, H, cache_length * Tg)
                        print(f"Tg={Tg} model_size={model_size} cache_length={cache_length * Tg}")
                        # warmup
                        for _ in range(4):
                            y = model_sampling(model, trainer, Tg)
                        
                        total_time = []
                        # timing module
                        for _ in range(8):
                            with PytorchTimer(verbose=False) as t:
                                y = model_sampling(model, trainer, Tg)
                            total_time.append(t.elapsed)
                        d[f"model_size={model_size} Tg={Tg} cache_length={0}"] = [np.mean(total_time), np.std(total_time)]

    print(d)
    df = pd.DataFrame(data=d, index=["runtime_mean(ms)", "runtime_std(ms)"])
    print(df)
    df.to_csv(f"logs/{today}-mem_demo-{model_size}_K={K}.csv")
print(time.time() - start)
# completion = ''.join([train_dataset.itos[int(i)] for i in y])
# print(completion)