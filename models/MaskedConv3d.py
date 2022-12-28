import torch
import torch.nn as nn
from torch.nn import functional as F

import math


class MaskedConv3d(nn.Module):
    def __init__(self, channels, kernel_size, stride, padding, bias=True):
        super().__init__()
        # Record the parameters
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define the weight and bias
        self.weight = nn.Parameter(torch.zeros(1, 1, *kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels)) if bias else None

        # Define the mask parameter
        self.register_buffer("mask", torch.ones_like(self.weight))
        kC, kH, kW = kernel_size
        self.mask[:, :, kC//2+1:, :, :] = 0
        self.mask[:, :, kC//2, kH//2+1:, :] = 0
        self.mask[:, :, kC//2, kH//2, kW//2:] = 0

        # Initialize the weight
        with torch.no_grad():
            self.weight.uniform_(-math.sqrt(1/(kC*kH*kW)), math.sqrt(1/(kC*kH*kW)))
            self.weight *= self.mask

    def forward(self, inputs):
        x = inputs.unsqueeze(1)
        y = F.conv3d(x, self.mask * self.weight, None, self.stride, self.padding)
        z = y.squeeze()
        outputs = z + self.bias.reshape(1, self.channels, 1, 1)

        return outputs


if __name__ == "__main__":
    tensor_dtypes = {"device": torch.device("cuda:0"), "dtype": torch.float32}
    model = MaskedConv3d(128, [5, 5, 5], 1, [2, 2, 2]).to(**tensor_dtypes)

    x = torch.zeros(8, 128, 16, 16, **tensor_dtypes)
    y = model(x)

    print(y.shape)
