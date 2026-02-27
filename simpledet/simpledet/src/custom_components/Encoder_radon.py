# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmdet.registry import MODELS
from typing import List, Tuple

class ConvNextV2Block(nn.Module):
    """
    ConvNextV2 Block implementation with Global Response Normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size for depthwise convolution.
        expansion_ratio (float): Expansion ratio for MLP.
        drop_path (float): Drop path rate for stochastic depth.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, 
                 expansion_ratio: float = 4.0, drop_path: float = 0.0):
        super().__init__()
        assert in_channels > 0 and out_channels > 0, 'Channel dimensions must be positive'
        assert kernel_size % 2 == 1, 'Kernel size must be odd'
        
        self.dwconv = nn.Conv2d(in_channels, in_channels, kernel_size, 
                               padding=kernel_size // 2, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels, eps=1e-6)
        
        hidden_dim = int(in_channels * expansion_ratio)
        self.pwconv1 = nn.Linear(in_channels, hidden_dim)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, out_channels)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_residual = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvNextV2 block."""
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        
        return residual + self.drop_path(x) if self.use_residual else x


class GlobalResponseNorm(nn.Module):
    """
    Global Response Normalization for ConvNextV2.

    Args:
        dim (int): Number of channels.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        assert dim > 0, 'Dimension must be positive'
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Global Response Normalization."""
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Args:
        drop_prob (float): Probability of dropping paths.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        assert 0.0 <= drop_prob <= 1.0, 'Drop probability must be between 0 and 1'
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply stochastic depth."""
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ChannelAttention(nn.Module):
    """
    Channel Attention Module with squeeze-and-excitation.

    Args:
        in_channels (int): Number of input channels.
        reduction (int): Reduction ratio for channel attention.
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        assert in_channels > 0, 'Input channels must be positive'
        assert reduction > 0, 'Reduction ratio must be positive'
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for clutter suppression.

    Args:
        kernel_size (int): Kernel size for spatial attention.
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial attention."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(out))


class ClutterSuppressionModule(nn.Module):
    """
    Advanced clutter suppression using multiple techniques.

    Args:
        in_channels (int): Number of input channels.
    """
    
    def __init__(self, in_channels: int):
        super().__init__()
        assert in_channels > 0, 'Input channels must be positive'
        
        self.channel_att = ChannelAttention(in_channels)
        self.spatial_att = SpatialAttention()
        self.morph_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU(inplace=True)
        self.freq_filter = nn.Parameter(torch.ones(1, in_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply clutter suppression techniques."""
        x = self.channel_att(x)
        x = self.spatial_att(x)
        x = self.act(self.norm(self.morph_conv(x)))
        return x * self.freq_filter


class FlexibleRadonTransform(nn.Module):
    """
    Flexible Radon Transform for sinogram generation.

    Args:
        num_angles (int): Number of projection angles.
    """
    
    def __init__(self, num_angles: int = 180):
        super().__init__()
        assert num_angles > 0, 'Number of angles must be positive'
        self.num_angles = num_angles
        self.register_buffer('angles', torch.linspace(0, np.pi, num_angles, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate sinograms using Radon transform."""
        B, C, H, W = x.shape
        grid_lin = torch.linspace(-1, 1, H, device=x.device)
        Y, X = torch.meshgrid(grid_lin, grid_lin, indexing='ij')
        base_grid = torch.stack((X, Y), dim=-1)

        sinograms = []
        for theta in self.angles:
            rot_matrix = torch.tensor([
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)]
            ], device=x.device)
            rot_grid = torch.matmul(base_grid, rot_matrix.T).unsqueeze(0).expand(B, -1, -1, -1)
            sampled = F.grid_sample(x, rot_grid, align_corners=True, mode='bilinear')
            sinograms.append(sampled.sum(dim=2))

        return torch.stack(sinograms, dim=2)


class RadonConvNextBlock(nn.Module):
    """
    ConvNextV2 block with integrated Radon transform.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_angles (int): Number of angles for Radon transform.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_angles: int = 90):
        super().__init__()
        assert in_channels > 0 and out_channels > 0, 'Channel dimensions must be positive'
        assert num_angles > 0, 'Number of angles must be positive'
        
        self.conv_block = ConvNextV2Block(in_channels, out_channels)
        self.radon_transform = FlexibleRadonTransform(num_angles)
        self.radon_proj = nn.Sequential(
            nn.Conv2d(num_angles, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Radon-ConvNext fusion."""
        conv_feat = self.conv_block(x)
        radon_out = self.radon_transform(conv_feat).transpose(1, 2)
        radon_feat = self.radon_proj(radon_out)
        radon_feat = F.adaptive_avg_pool2d(radon_feat, conv_feat.shape[-2:])
        fused = torch.cat([conv_feat, radon_feat], dim=1)
        return self.fusion(fused)


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
                 in_channels: int = 3,
                 base_channels: int = 64,
                 depths: List[int] = [2, 2, 6, 2],
                 num_angles: int = 90,
                 drop_path_rate: float = 0.1,
                 out_indices: Tuple[int, ...] = (1, 2, 3),
                 frozen_stages: int = -1):
        super().__init__()
        
        # Validation
        assert in_channels > 0, 'Input channels must be positive'
        assert base_channels > 0, 'Base channels must be positive'
        assert len(depths) > 0, 'Depths list cannot be empty'
        assert all(d > 0 for d in depths), 'All depth values must be positive'
        assert len(depths) >= max(out_indices) + 1, 'depths must be longer than max out_indices'
        assert num_angles > 0, 'Number of angles must be positive'
        assert 0.0 <= drop_path_rate <= 1.0, 'Drop path rate must be between 0 and 1'
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.depths = depths
        self.num_angles = num_angles
        self.drop_path_rate = drop_path_rate
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        
        # Channel dimensions for each stage
        self.stage_channels = [base_channels * (2 ** i) for i in range(len(depths))]
        
        # Stem layer
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, stride=4),
            nn.BatchNorm2d(base_channels)
        )
        
        # Initialize branches
        self._build_branch_a()
        self._build_branch_b()
        self._build_branch_c()
        
        # Feature fusion and output projection
        self._build_fusion_layers()
        
        # Initialize weights
        self.init_weights()
        
        # Freeze stages if specified
        self._freeze_stages()

    def _build_branch_a(self) -> None:
        """Build Branch A: Pure ConvNextV2 feature extractors."""
        self.branch_a = nn.ModuleList()
        
        dp_rates = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]
        cur = 0
        
        for i, depth in enumerate(self.depths):
            stage = nn.ModuleList()
            
            # Downsample layer
            if i > 0:
                downsample = nn.Sequential(
                    nn.BatchNorm2d(self.stage_channels[i-1]),
                    nn.Conv2d(self.stage_channels[i-1], self.stage_channels[i], 2, stride=2)
                )
                stage.append(downsample)
            
            # ConvNextV2 blocks
            for j in range(depth):
                in_ch = self.stage_channels[i] if i > 0 or j > 0 else self.base_channels
                block = ConvNextV2Block(in_ch, self.stage_channels[i], drop_path=dp_rates[cur + j])
                stage.append(block)
            
            self.branch_a.append(stage)
            cur += depth

    def _build_branch_b(self) -> None:
        """Build Branch B: Radon layers mixed with ConvNextV2."""
        self.branch_b = nn.ModuleList()
        
        for i, depth in enumerate(self.depths):
            stage = nn.ModuleList()
            
            # Downsample layer
            if i > 0:
                downsample = nn.Sequential(
                    nn.BatchNorm2d(self.stage_channels[i-1]),
                    nn.Conv2d(self.stage_channels[i-1], self.stage_channels[i], 2, stride=2)
                )
                stage.append(downsample)
            
            # Radon-ConvNext blocks
            for j in range(depth):
                in_ch = self.stage_channels[i] if i > 0 or j > 0 else self.base_channels
                block = RadonConvNextBlock(in_ch, self.stage_channels[i], self.num_angles)
                stage.append(block)
            
            self.branch_b.append(stage)

    def _build_branch_c(self) -> None:
        """Build Branch C: Channel attention with clutter suppression."""
        self.branch_c = nn.ModuleList()
        
        for i, depth in enumerate(self.depths):
            stage = nn.ModuleList()
            
            # Downsample layer
            if i > 0:
                downsample = nn.Sequential(
                    nn.BatchNorm2d(self.stage_channels[i-1]),
                    nn.Conv2d(self.stage_channels[i-1], self.stage_channels[i], 2, stride=2)
                )
                stage.append(downsample)
            
            # Clutter suppression blocks
            for j in range(depth):
                in_ch = self.stage_channels[i] if i > 0 or j > 0 else self.base_channels
                
                # ConvNext block followed by clutter suppression
                conv_block = ConvNextV2Block(in_ch, self.stage_channels[i])
                clutter_block = ClutterSuppressionModule(self.stage_channels[i])
                
                stage.extend([conv_block, clutter_block])
            
            self.branch_c.append(stage)

    def _build_fusion_layers(self) -> None:
        """Build feature fusion layers for multi-level output."""
        self.fusion_layers = nn.ModuleDict()
        
        # Target output channels: 128, 256, 512
        target_channels = [128, 256, 512]
        
        for i, out_idx in enumerate(self.out_indices):
            # Each branch contributes features
            input_channels = self.stage_channels[out_idx] * 3  # 3 branches
            output_channels = target_channels[i] if i < len(target_channels) else target_channels[-1]
            
            fusion_layer = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True)
            )
            
            self.fusion_layers[f'fusion_{out_idx}'] = fusion_layer

    def _freeze_stages(self) -> None:
        """Freeze specified stages."""
        if self.frozen_stages >= 0:
            # Freeze stem
            for param in self.stem.parameters():
                param.requires_grad = False
            
            # Freeze stages
            for i in range(self.frozen_stages + 1):
                if i < len(self.branch_a):
                    for param in self.branch_a[i].parameters():
                        param.requires_grad = False
                    for param in self.branch_b[i].parameters():
                        param.requires_grad = False
                    for param in self.branch_c[i].parameters():
                        param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass through the multi-branch backbone.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].

        Returns:
            List[torch.Tensor]: List of multi-level feature maps with channels [128, 256, 512].
        """
        # Stem processing
        x = self.stem(x)
        
        # Initialize branch outputs
        branch_a_out = x
        branch_b_out = x
        branch_c_out = x
        
        outputs = []
        
        # Process through each stage
        for stage_idx in range(len(self.depths)):
            # Branch A processing
            for layer in self.branch_a[stage_idx]:
                branch_a_out = layer(branch_a_out)
            
            # Branch B processing
            for layer in self.branch_b[stage_idx]:
                branch_b_out = layer(branch_b_out)
            
            # Branch C processing
            for layer in self.branch_c[stage_idx]:
                branch_c_out = layer(branch_c_out)
            
            # Collect outputs at specified indices
            if stage_idx in self.out_indices:
                # Ensure all branches have the same spatial dimensions
                target_size = branch_a_out.shape[-2:]
                branch_a_resized = F.adaptive_avg_pool2d(branch_a_out, target_size)
                branch_b_resized = F.adaptive_avg_pool2d(branch_b_out, target_size)
                branch_c_resized = F.adaptive_avg_pool2d(branch_c_out, target_size)
                
                # Concatenate features from all branches
                fused_features = torch.cat([branch_a_resized, branch_b_resized, branch_c_resized], dim=1)
                
                # Apply fusion layer
                fusion_key = f'fusion_{stage_idx}'
                if fusion_key in self.fusion_layers:
                    fused_features = self.fusion_layers[fusion_key](fused_features)
                
                outputs.append(fused_features)
        
        return outputs

    def init_weights(self) -> None:
        """Initialize weights for the backbone."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


