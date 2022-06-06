#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import copy
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):

    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for i in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x, pos_embed, qk_mask=None, k_mask=None):
        for i, l in enumerate(self.layers):
            x = l(x, pos_embed, qk_mask, k_mask)
        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        embed_dim,
        ffn_dim,
        self_attn_cls,
        ffn_cls,
        n_head=8,
        dropout=0.1,
        drop_fn=nn.Dropout,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.self_attn = self_attn_cls(embed_dim,
                                       n_head=n_head,
                                       dropout=dropout,
                                       drop_fn=drop_fn)
        self.ffn = ffn_cls(embed_dim,
                           ffn_dim=ffn_dim,
                           dropout=dropout,
                           drop_fn=drop_fn)

    def forward(self, x, pos_embed, qk_mask=None, k_mask=None):
        x, _ = self.self_attn(
            qkv=x,
            qpos=pos_embed,
            kpos=pos_embed,
            qk_mask=qk_mask,
            k_mask=k_mask,
        )
        x = self.ffn(x)
        return x
