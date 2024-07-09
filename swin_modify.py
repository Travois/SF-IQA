

from typing import Any, List

import torch.nn as nn
from torch import Tensor



class Swin(nn.Module):

    def __init__(
        self,
        name:str,
        pretrained:bool
    ) -> None:
        super(Swin, self).__init__()
        from timm.models.swin_transformer_v2 import (swinv2_tiny_window8_256,
                                                     swinv2_tiny_window16_256)
        if name == 'swinv2_tiny_patch4_window8_256':
            swintransformer=swinv2_tiny_window8_256(pretrained=pretrained)
        elif name =='swinv2_tiny_patch4_window16_256':
            swintransformer=swinv2_tiny_window16_256(pretrained=pretrained)
        self.patch_embed = swintransformer.patch_embed
        self.pos_drop = swintransformer.pos_drop
        self.layers = swintransformer.layers
        self.norm = swintransformer.norm
        self.head = swintransformer.head

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        b,c,w,h=x.shape                         #8 3 256 256
        x=self.patch_embed(x)                   #8 64 64 96
        x=self.pos_drop(x)
        x = self.layers[0](x)                   #8 64 64 96
        l1 = x.view(b, c*8*8, w//8, h//8)                                  #b w/4 h/4 c*4*4*2
        x = self.layers[1](x)                   #8 32 32 192
        l2 = x.view(b, c*8*8*2, w//16, h//16)                                  #b w/8 h/8 c*4*4*4
        x = self.layers[2](x)                   #8 16 16 384
        l3 = x.view(b, c*8*8*4, w//32, h//32)                                  #b w/16 h/16 c*4*4*8
        x = self.layers[3](x)                   #8 8 8 768
        l4 = x.view(b, c*8*8*4, w//32, h//32)                                  #b w/32 h/32 c*4*4*16
        x = self.norm(x)                        #8 8 8 768
        x = self.head(x)                        #8 1000

        return x,l1,l2,l3,l4

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _swin(
    name: str,
    pretrained: bool,
) -> Swin:
    model = Swin(name, pretrained)
    return model

def swinv2_tiny_patch4_window8_256(pretrained: bool = True) -> Swin:
    r"""swinv2_tiny_patch4_window8_256 model 
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _swin('swinv2_tiny_patch4_window8_256', pretrained)

def swinv2_tiny_patch4_window16_256(pretrained: bool = True) -> Swin:
    r"""swinv2_tiny_patch4_window16_256 model 
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _swin('swinv2_tiny_patch4_window16_256', pretrained)
