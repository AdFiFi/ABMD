import torch
import torchvision
from torch import nn
from .base import ModelConfig


class UNetConfig(ModelConfig):
    def __init__(self, num_classes):
        super().__init__(num_classes=num_classes,
                         d_hidden=2048)


class UNet(nn.Module):
    def __init__(self, config: UNetConfig):
        super().__init__()
        self.unet = torchvision.models.vit_b_16

import transformers

m = transformers.ViTModel