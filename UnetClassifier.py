from typing import Optional
import torch
import torch.nn as nn
from torch.autograd import Variable

from segmentation_models_pytorch.base import ClassificationHead
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import initialization as init


class UnetClassifier(nn.Module):
    def __init__(self, encoder_name='timm-efficientnet-b0', encoder_depth=5, encoder_weights='imagenet',
                 in_channels=3, num_classes=3):
        super(UnetClassifier, self).__init__()

        # Encoder half of unet
        self.encoder = get_encoder(encoder_name, in_channels=in_channels, depth=encoder_depth, weights=encoder_weights,)
        self.num_classes = num_classes

        # Parameters for the classification head - at some point we might want these as class attributes
        # Currently, all are their default's except activation
        aux_params = {
            'pooling': 'avg',
            'dropout': 0.0,
            'activation': None,
            'classes': num_classes
        }
        self.classification_head = ClassificationHead(in_channels=self.encoder.out_channels[-1], **aux_params)
        init.initialize_head(self.classification_head)

    def forward(self, x):
        features = self.encoder(x)
        logits = self.classification_head(features[-1])

        return logits
