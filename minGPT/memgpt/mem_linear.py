from utils import check_shape, CachedModule, PytorchTimer
import torch
import torch.nn as nn
import logging
# from pthflops import count_ops
logging.basicConfig(level=logging.DEBUG)


# print(device)

class CachedLinear(CachedModule):
    """ cached nn.Linear layer """
    def __init__(self, in_features, out_features, bias=True, x=None, **kwargs):
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()

        CachedModule.__init__(self, x)
        self.layer = nn.Linear(in_features, out_features, bias, **kwargs, device=self.device)

    def forward(self, x, i=0):
        """ 
        x: BTH 
        cache: B(T-1)H  
        new_out: B*1*H
        """
        B, T, _ = x.shape
        if self.cache is not None:
            cache = check_shape(self.cache, (B, T - 1 + i, self.layer.out_features))
            new_out = check_shape(self.layer(x[:, -1:, :]), (B, 1, self.layer.out_features))
            y = torch.cat([cache, new_out], dim=1)  
        else:
           y = check_shape(self.layer(x), (B, T + i, self.layer.out_features))
        self.set_cache(y)
        return y


if __name__ == "__main__":
    B, T, H = (16, 128, 768)

    layer = CachedLinear(H, H, False)
    
    layer.clear_cache()
    x = torch.randn((B, T, H))
    with PytorchTimer(verbose=True):
        y = check_shape(layer(x), (B, T, H))

    # layer.clear_cache()
    # x = torch.randn((B, T, H))
    # with PytorchTimer(verbose=True):
    #     y = check_shape(layer(x), (B, T, H))

    # layer.clear_cache()
    # x = torch.randn((B, T, H))
    # with PytorchTimer(verbose=True):
    #     y = check_shape(layer(x), (B, T, H))

    # layer.clear_cache()
    # x = torch.randn((B, T, H))
    # with PytorchTimer(verbose=True):
    #     y = check_shape(layer(x), (B, T, H))

    # logging.debug(f"test cache")
    # for i in range(1, 10):
    #     x = torch.randn((B, T + i, H))
    #     with PytorchTimer(verbose=True):
    #         y = check_shape(layer(x), (B, T + i, H))
