import torch
import torchvision
from torch import nn
from .base import *
from transformers import \
    (ResNetConfig as Config,
     ResNetModel)


class ResNetConfig(Config, VisionEncoderConfig):
    def __init__(self, num_classes, num_channels):
        super().__init__(num_labels=num_classes,
                         num_channels=num_channels)
        VisionEncoderConfig.__init__(self,
                                     num_classes=num_classes,
                                     d_hidden=self.hidden_sizes[-1])


#
# class ResNet(nn.Module):
#     def __init__(self, config: ResNetConfig):
#         super().__init__()
#         self.resnet = torchvision.models.resnet50(num_classes=config.num_classes)
#
#     def forward(self, x):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)
#
#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)
#
#         x = self.resnet.avgpool(x)
#         hidden = torch.flatten(x, 1)
#         x = self.resnet.fc(hidden)
#
#         return x, hidden


class ResNet(nn.Module):
    def __init__(self, config: ResNetConfig):
        super().__init__()
        self.resnet = ResNetModel(config)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(config.hidden_sizes[-1],
                                    config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, pixel_values, **kwargs):
        outputs = self.resnet(pixel_values, return_dict=True)
        pooled_output = self.flatten(outputs.pooler_output)
        logits = self.classifier(pooled_output)
        return VisionEncoderOutputs(logits=logits,
                                    vision_state=pooled_output)
