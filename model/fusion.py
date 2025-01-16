import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
import math
import numpy as np
import torch.distributed as dist
from .layers import conv_layer, deconv_layer
import os
from functools import partial



class Fusion(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                 dino_layers= 12,
                 output_dinov2 =[4, 8] ,
                ):
        super().__init__()

        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.dino_layers = dino_layers
        self.output_dinov2 = output_dinov2
        self.n_ctx_visual = 0

        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)
        nn.init.normal_(textual_ctx_vectors, std=0.02)
        self.initialize_parameters()

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, img, text, txt_backbone,dino):
        B=img.shape[0]
        img = img.type(txt_backbone.dtype)
        vis_outs = []
        outputs=[]
        txt = txt_backbone.token_embedding(text).type(
            txt_backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = txt_backbone.transformer
        txt = txt + txt_backbone.positional_embedding.type(txt_backbone.dtype)[:txt.size(1)]
        txt = txt.permute(1, 0, 2)  # BLD -> LBD
        
        #dinov2  
        net_input = img.clone()
        B, nc, w, h = net_input.shape
        dino_f = dino.patch_embed(net_input)
        dino_f = torch.cat((dino.cls_token.expand(dino_f.shape[0], -1, -1), dino_f), dim=1)
        dino_f = dino_f + dino.interpolate_pos_encoding(dino_f, w, h)
        dino_f = torch.cat(
            (
                dino_f[:, :1],
                dino.register_tokens.expand(dino_f.shape[0], -1, -1),
                dino_f[:, 1:],
            ),
            dim=1,
        )
        features_dino=[]
        for i in range(self.num_layers):
            txt = txt_enc.resblocks[i](txt)

        # language
        txt = txt.permute(1, 0, 2)  # LBD -> BLD
        txt = txt_backbone.ln_final(txt).type(txt_backbone.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt[torch.arange(txt.shape[0]),
                  text.argmax(dim=-1)] @ txt_backbone.text_projection# get sentence-level feature Fs
        
        for i in range(self.dino_layers):
            dino_f = dino.blocks[i](dino_f, txt)
            if i in self.output_dinov2:
                features_dino.append(dino_f)
        
        dino_f = dino.norm(dino_f)
        features_dino.append(dino_f)
        

        for i, feature_dino in enumerate(features_dino):
            feature_dino=feature_dino[:, 4 + 1 :]
            B,L,C = feature_dino.shape
            H = int(L ** 0.5)
            W = L // H
            feature_dino = feature_dino.reshape(B, H, W, C).permute(0, 3, 1, 2)

            vis_outs.append(feature_dino)
 

        # forward

        output = vis_outs , txt, state

        return output



