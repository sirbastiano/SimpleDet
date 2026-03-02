# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmdet.registry import MODELS
from typing import List
import timm



@MODELS.register_module()
class Backbone(nn.Module):
    """
    Multi-branch backbone with ConvNextV2, Radon, and Clutter suppression branches.

    Args:
        in_channels (int): Number of input channels.
        base_channels (int): Base number of channels.
        depths (List[int]): Number of blocks in each stage.
        num_angles (int): Number of angles for Radon transform.
        drop_path_rate (float): Drop path rate for regularization.
        out_indices (Tuple[int, ...]): Output feature indices.
        frozen_stages (int): Number of frozen stages.
    """
    
    def __init__(self, 
                encoder_name: str = 'convnextv2_huge.fcmae_ft_in22k_in1k_512',
                pretrained: bool = True,                
                features_only: bool = True,
                ):
        super().__init__()
        

        self.encoder = timm.create_model(
            encoder_name,
            pretrained=pretrained,
            features_only=features_only,
        )
        
        
  
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the multi-branch backbone.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            List[torch.Tensor]: List of multi-level feature maps with channels.
        """
        return self.encoder(x)

    