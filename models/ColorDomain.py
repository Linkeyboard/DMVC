import torch
import torch.nn as nn
from torch.nn import functional as F


class RGBtoYUV420(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb_to_yuv = nn.Parameter(torch.Tensor(
            [+0.299, +0.587, +0.114],
            [-0.147, +0.289, +0.436],
            [+0.615, -0.515, -0.100],
        ).reshape(3, 3, 1, 1))

    def forward(self, inputs):
        outputs = F.conv2d(inputs, self.rgb_to_yuv)

        return outputs
