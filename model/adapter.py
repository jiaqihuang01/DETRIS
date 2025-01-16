from typing import Tuple, Union, List, Any
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class TextAdapter(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        skip_connect=False,
    ) -> None:
        super().__init__()
        self.skip_connect = skip_connect
        conv_block = BasicConv1d
        self.dense_branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.dense_branch2 = nn.Sequential(
            conv_block(in_channels + ch1x1, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.dense_branch3 = nn.Sequential(
            conv_block(in_channels + ch1x1 + ch3x3, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )
        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

    def forward(self, x: Tensor) -> List[Tensor]:
        x0 = self.D_fc1(x)
        B, P, D = x0.shape

        x0 = F.relu(x0, inplace=True)

        xs = x0[:, 1:, :].permute(0, 2, 1)  
        dense_branch1 = self.dense_branch1(xs)
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))
        dense_branch3 = self.dense_branch3(torch.cat([xs, dense_branch1, dense_branch2], dim=1))
        outputs = [dense_branch1, dense_branch2, dense_branch3]
        outputs = torch.cat(outputs, dim=1).permute(0, 2, 1) 

        clstoken = x0[:, 0:1, :]
        outputs = torch.cat([clstoken, outputs], dim=1)

        outputs += x0

        outputs = self.D_fc2(outputs)

        if self.skip_connect:
            outputs += x
        return outputs
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=True, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
    
class CrossModalAttention(nn.Module):
    def __init__(self, image_dim, text_dim, embed_dim, num_heads, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        
        self.image_proj = nn.Linear(image_dim, embed_dim)
        self.text_proj = nn.Linear(text_dim, embed_dim)

        self.back_proj = nn.Linear(embed_dim, image_dim)
        
    def forward(self, image_features, text_features, attention_mask=None):


        image_features = self.image_proj(image_features)
        text_features = self.text_proj(text_features)

        query = image_features.permute(1, 0, 2)
        key = text_features.permute(1, 0, 2)
        value = text_features.permute(1, 0, 2)
        
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attention_mask)
        
        attn_output = self.back_proj(attn_output)
        
        return attn_output.permute(1, 0, 2)


class DenseAligner(nn.Module):
    def __init__(
        self,
        fc_in_channels: int,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        skip_connect=False,
        embed_dim=128,
        num_heads=8,
        text_dim=512,
    ) -> None:
        super().__init__()
        self.skip_connect=skip_connect
        conv_block = BasicConv2d
        self.dense_branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.dense_branch2 = nn.Sequential(
            conv_block(in_channels+ch1x1, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.dense_branch3 = nn.Sequential(
            conv_block(in_channels+ch1x1+ch3x3, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=5, padding=2),
        )

        self.D_fc1 = nn.Linear(fc_in_channels, in_channels)
        self.D_fc2 = nn.Linear(in_channels, fc_in_channels)

        self.cross = CrossModalAttention(in_channels, text_dim, embed_dim, num_heads)

        self._initialize_weights()

    def _initialize_weights(self):
   
        for module in self.modules():
            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)  
            elif isinstance(module, nn.Linear):
  
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)        

    def forward(self, x: Tensor, text_features,split_token=5) -> List[Tensor]:

        x0 = self.D_fc1(x)
        B,P,D = x0.shape
        W = H = int(math.sqrt(P-1))

        x0 = F.relu(x0, inplace=True)
        
        xs = x0[:,split_token:,:]
        # B, W, H, D / 8, 32, 32, 384 -> B, D, W, H / B, 384, 32, 32
        xs = xs.reshape(B,W,H,D).permute(0,3,1,2)
        # B, 192, 32, 32
        dense_branch1 = self.dense_branch1(xs)
        # B, 96, 32, 32
        dense_branch2 = self.dense_branch2(torch.cat([xs, dense_branch1], dim=1))
        # B, 96, 32, 32
        dense_branch3 = self.dense_branch3(torch.cat([xs, dense_branch1, dense_branch2], dim=1))
        outputs = [dense_branch1, dense_branch2, dense_branch3]
        # B, 384, 32, 32
        outputs = torch.cat(outputs,dim=1) + xs
        # B, 384, 32, 32 -> B, 384, 1024 -> B, 1024, 384
        outputs = outputs.reshape(B,D,W*H).permute(0,2,1)
        # text fusion
        outputs = self.cross(outputs, text_features)

        clstoken =  x0[:,0:split_token,:]
        outputs = torch.cat([clstoken,outputs],dim=1)

        outputs += x0

        outputs = self.D_fc2(outputs)
        if self.skip_connect:
            outputs+=x
        return outputs