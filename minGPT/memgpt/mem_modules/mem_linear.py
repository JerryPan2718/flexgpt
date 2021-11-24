from utils import check_shape, CachedModule, PytorchTimer
import torch
import torch.nn as nn
from loguru import logger

class CachedLinear(CachedModule):
    """ cached nn.Linear layer """
    def __init__(self, in_features, out_features, bias, x=None, **kwargs):
        CachedModule.__init__(self, x)
        self.layer = nn.Linear(in_features, out_features, bias, **kwargs)

    def forward(self, x):
        """ 
        x: BTH 
        cache: B(T-1)H  
        new_out: B*1*H
        """
        B, T, _ = x.shape
        if self.cache is not None:
            cache = check_shape(self.cache, (B, T - 1, self.layer.out_features))
            new_out = check_shape(self.layer(x[:, -1:, :]), (B, 1, self.layer.out_features))
            y = torch.cat([cache, new_out], dim=1)  
        else:
           y = check_shape(self.layer(x), (B, T, self.layer.out_features))
        self.set_cache(y)
        return y


if __name__ == "__main__":
    B, T, H = (16, 128, 768)

    layer = CachedLinear(H, H, False)
    
    layer.clear_cache()
    x = torch.randn((B, T, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T, H))

    layer.clear_cache()
    x = torch.randn((B, T, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T, H))

    layer.clear_cache()
    x = torch.randn((B, T, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T, H))

    layer.clear_cache()
    x = torch.randn((B, T, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T, H))

    logger.debug(f"test cache")
    x = torch.randn((B, T + 1, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T + 1, H))

    x = torch.randn((B, T + 2, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T + 2, H))

    x = torch.randn((B, T + 3, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T + 3, H))