import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_

import math


class MaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        # Record the parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define the weight and bias
        self.weight = nn.Parameter(torch.zeros(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Define the mask parameter
        self.register_buffer("mask", torch.ones_like(self.weight))
        kH, kW = kernel_size, kernel_size
        self.mask[:, :, kH//2+1:, :] = 0
        self.mask[:, :, kH//2, kW//2:] = 0

        # Initialize the weight
        with torch.no_grad():
            xavier_uniform_(self.weight)
            self.weight *= self.mask

    def forward(self, inputs):
        return F.conv2d(inputs, self.mask * self.weight, self.bias, self.stride, self.padding)
