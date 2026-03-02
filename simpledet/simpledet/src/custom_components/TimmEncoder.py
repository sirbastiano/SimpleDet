# Copyright (c) OpenMMLab. All rights reserved.

import torch.nn as nn
from mmdet.registry import MODELS
import timm 

@MODELS.register_module()
class TimmEncoder(nn.Module):

    def __init__(self, model_name, features_only=True, pretrained=True, in_chans=1, frozen_stages=1):
        super(TimmEncoder, self).__init__()
        # Instantiate the model
        self.model = timm.create_model(model_name, features_only=features_only, pretrained=pretrained, in_chans=in_chans)
        
        # Freeze the specified number of stages
        if frozen_stages > 0:
            for idx, item in enumerate(self.model.children()):
                if idx < frozen_stages:
                    for param in item.parameters():
                        param.requires_grad = False
                else:
                    break        

    def forward(self, x):  # should return a tuple
        return tuple(self.model(x))
    
