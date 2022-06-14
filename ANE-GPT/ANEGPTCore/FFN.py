#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn
from ANEGPTCore.Layer_Normal import LayerNormANE


class FFN(nn.Module):
    """
     Feedforward neural network layers
    """
    def __init__(self, embed_dim, ffn_dim, dropout=0.1, **kwargs):
        super(FFN,self).__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(embed_dim, ffn_dim, 1),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0. else nn.Identity(),# 对参数不敏感的占位符标识运算符。
            nn.Conv2d(ffn_dim, embed_dim, 1),
        ])

    # 定义前向传播的“暗示”规则
    def _forward_impl(self, x, **kwargs):
        for l in self.layers:
            x = l(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)# 调用_forward_impl函数规则用于前向传播




# 残差前馈网络
class ResidualFFN(FFN):
    def __init__(self, embed_dim, dropout=0.1, drop_fn=nn.Dropout, **kwargs):
        super(ResidualFFN,self).__init__(embed_dim, dropout=dropout, **kwargs)

        self.rdropout = drop_fn(dropout) if dropout > 0. else nn.Identity()
        self.rnorm = LayerNormANE(embed_dim)

    def forward(self, x):
        residual = self._forward_impl(x)
        return self.rnorm(self.rdropout(residual) + x)


# 定义预归一化残差前馈神经网络层
class PreNormResidualFFN(ResidualFFN):
    def forward(self, x):
        residual = self.rdropout(self._forward_impl(self.rnorm(x)))
        return x + residual
