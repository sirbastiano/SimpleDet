# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
from mmdet.registry import MODELS
import timm 

@MODELS.register_module()
class TimmEncoder(nn.Module):

    def __init__(self, model_name, features_only=True, pretrained=True, in_chans=1):
        super(TimmEncoder, self).__init__()
        # Instantiate the model
        self.model = timm.create_model(model_name, features_only=features_only, pretrained=pretrained, in_chans=in_chans)        

    def forward(self, x):  # should return a tuple
        return tuple(self.model(x))