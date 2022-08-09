#
# For licensing see accompanying LICENSE.md file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#
import torch
from torch import nn


class LayerNormANE(nn.Module):
    """ LayerNorm optimized for Apple Neural Engine (ANE) execution

    Note: This layer only supports normalization over the final dim. It expects `num_channels`
    as an argument and not `normalized_shape` which is used by `torch.nn.LayerNorm`.
    """
    def __init__(self,num_channels,
                 clip_mag=None,
                 eps=1e-5,
                 elementwise_affine=True):
        """
        Args:
            num_channels:       预期输入数据格式为 BC1S 的通道数 (C)。 S代表序列长度。
            clip_mag:           在应用层规范之前，用于限制输入范围的可选浮点值。
                                如果指定，有助于降低溢出风险。
            eps:                避免被零除的小值
            elementwise_affine: 如果为 True，则添加可学习的通道移位（偏差）和尺度（权重）参数
        """
        super(LayerNormANE, self).__init__()
        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        # 一般来说，Transformer 架构处理一个 3D 输入张量，该输入张量包含一批 B 序列的 S 维数为 C 的嵌入向量。我们以 (B, C, 1, S)
        # 数据格式表示此张量，因为最有利于ANE（硬件和软件堆栈）是 4D 和通道优先的。
        self.expected_rank = len('BC1S') # 为了便于ANE的运行加速进行定义BC1S的shape的tensor,因此这里是预期的tensor的索引数为了便于下文定义和判断
        self.num_channels = num_channels
        self.eps = eps
        self.clip_mag = clip_mag
        self.elementwise_affine = elementwise_affine


        if self.elementwise_affine: # 添加可学习的通道（逐元素仿射）
            self.weight = nn.Parameter(torch.Tensor(num_channels))
            self.bias = nn.Parameter(torch.Tensor(num_channels))

        self._reset_parameters() #


    # 初始化权重参数
    def _reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)
    # 定义网络的传播
    def forward(self, inputs):
        input_rank = len(inputs.size())
        # Principle 1: Picking the Right Data Format (machinelearning.apple.com/research/apple-neural-engine)
        # 将数据格式从BSC迁移到BC1S（最有利于ANE）
        if input_rank == 3 and inputs.size(2) == self.num_channels:
            inputs = inputs.transpose(1, 2).unsqueeze(2) # 将张量进行转置
            input_rank = len(inputs.size())

        assert input_rank == self.expected_rank
        assert inputs.size(1) == self.num_channels

        if self.clip_mag is not None:# 防止溢出风险设置(-clip_mag......,.....clip_mag+)范围的数值
            inputs.clamp_(-self.clip_mag, self.clip_mag)

        # 机算通道的mean
        channels_mean = inputs.mean(dim=1, keepdims=True)
        zero_mean = inputs - channels_mean
        zero_mean_sq = zero_mean ** 2 #  zero_mean * zero_mean
        denom = (zero_mean_sq.mean(dim=1, keepdims=True) + self.eps).rsqrt()


        out = zero_mean * denom

        if self.elementwise_affine:
            z = (out + self.bias.view(1, self.num_channels, 1, 1))
            x = self.weight.view(1, self.num_channels, 1, 1)

            out = z * x
        return out

