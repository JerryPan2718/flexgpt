from utils import check_shape, CachedModule, PytorchTimer, check_device_on_cuda
from mem_selfattn import CachedSelfAttn
from mem_block import MemBlock
import torch
import torch.nn as nn
from torch.nn import functional as F
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MemGPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, B=12, K=12, H=768, T=2048, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.B = B
        self.K = K
        self.H = H
        self.T = T

        for k,v in kwargs.items():
            setattr(self, k, v)

class MemGPT1Config(MemGPTConfig):
    """ GPT-1 like network roughly 125M params """
    B = 12
    K = 12
    H = 768

class MemGPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.T = self.config.T

        # input embedding stem
        self.tok_emb = nn.Embedding(self.config.vocab_size, self.config.H)
        # self.pos_emb = nn.Parameter(torch.zeros(1, self.config.B, self.config.H))
        self.pos_emb = nn.Parameter(torch.zeros(1, self.config.T, self.config.H))
        self.drop = nn.Dropout(self.config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[MemBlock(self.config) for _ in range(self.config.B)])
        # decoder head
        self.ln_f = nn.LayerNorm(self.config.H)
        self.head = nn.Linear(self.config.H, self.config.vocab_size, bias=False)

        self.block_size = config.B
        self.apply(self._init_weights)
        self.B_idx = 0

        logging.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        b, t = idx.size()
        # assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        if str(idx.device)[:4] != "cuda":
            print(f"idx.device: {idx.device}")
        
        if targets != None:
            print(f"targets: {targets}")
       
        # forward the GPT model
        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        # print(f"token_embeddings: {token_embeddings.shape}, position_embeddings: {position_embeddings.shape}")
        x = self.drop(token_embeddings + position_embeddings)
        # print(f"before blocks: {x.shape}")
        x = self.blocks(x)
        # print(f"after blocks: {x.shape}")
        x = self.ln_f(x)
        logits = self.head(x)


        # B, T, H = x.shape
        # y_new = torch.randn((B, 1, H), device=device)
        # y = x
        # x = check_shape(torch.cat((y, y_new), dim=-2), (B, T + 1, H))

        # self.B_idx += 1
        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
