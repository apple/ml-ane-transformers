#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
import torch.nn as nn

from .layer_norm import LayerNormANE


class FFN(nn.Module):

    def __init__(self, embed_dim, ffn_dim, dropout=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(embed_dim, ffn_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),
            nn.Conv2d(ffn_dim, embed_dim, 1),
        ])

    def _forward_impl(self, x, **kwargs):
        for l in self.layers:
            x = l(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


class ResidualFFN(FFN):

    def __init__(self, embed_dim, dropout=0.1, drop_fn=nn.Dropout, **kwargs):
        super().__init__(embed_dim, dropout=dropout, **kwargs)
        self.rdropout = drop_fn(dropout) if dropout > 0. else nn.Identity()
        self.rnorm = LayerNormANE(embed_dim)

    def forward(self, x):
        residual = self._forward_impl(x)
        return self.rnorm(self.rdropout(residual) + x)


class PreNormResidualFFN(ResidualFFN):

    def forward(self, x):
        residual = self.rdropout(self._forward_impl(self.rnorm(x)))
        return x + residual
