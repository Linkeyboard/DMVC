import math

import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from .quantize import quantize
from .math_ops import lower_bound


class CDF(nn.Module):
    def __init__(self, channels, filters):
        super(CDF, self).__init__()
        self._ch = int(channels)
        self._ft = (1,) + tuple(int(nf) for nf in filters) + (1,)

        # self._ft: (1, 3, 3, 3, 1)

        self._matrices = nn.ParameterList()
        self._biases = nn.ParameterList()
        self._factors = nn.ParameterList()

        init_scale = 10
        scale = init_scale ** (1 / (len(self._ft) - 1))
        for i in range(len(self._ft) - 1):
            # Define the matrices
            init = math.log(math.expm1(1 / scale / self._ft[i + 1]))
            matrix = torch.zeros(self._ch, self._ft[i + 1], self._ft[i]).fill_(init)
            matrix = matrix + torch.zeros_like(matrix).uniform_(-abs(init)/5, abs(init)/5)
            matrix = nn.Parameter(matrix)
            self._matrices.append(matrix)

            # Define the biases
            bias = nn.Parameter(torch.zeros(self._ch, self._ft[i + 1], 1).uniform_(-0.5, 0.5))
            self._biases.append(bias)

            # Define the factor
            if i < len(self._ft) - 2:
                factor = nn.Parameter(torch.zeros(self._ch, self._ft[i + 1], 1).fill_(0))
                self._factors.append(factor)

    def forward(self, inputs, stop_gradient):
        # Inputs shape => [Channels, 1, Values]
        logits = inputs

        for i in range(len(self._ft) - 1):
            matrix = self._matrices[i]
            if stop_gradient:
                matrix = matrix.detach()
            matrix = nn.functional.softplus(matrix)
            logits = torch.matmul(matrix, logits)

            bias = self._biases[i]
            if stop_gradient:
                bias = bias.detach()
            logits += bias

            if i < len(self._ft) - 2:
                factor = self._factors[i]
                if stop_gradient:
                    factor = factor.detach()
                factor = torch.tanh(factor)
                logits += factor * torch.tanh(logits)

        return logits


class FullFactorizedModel(nn.Module):
    def __init__(self, channels, filters, likelihood_bound):
        super(FullFactorizedModel, self).__init__()
        self._ch = channels
        self._cdf = CDF(self._ch, filters)
        self._likelihood_bound = likelihood_bound

        # Define the "optimize_integer_offset".
        self.register_parameter("quantiles", nn.Parameter(torch.zeros(self._ch, 1, 1)))
        self.register_buffer("target", torch.zeros(self._ch, 1, 1))

    def forward(self, inputs):
        # Reshape the inputs.
        inputs = torch.transpose(inputs, 0, 1)
        shape = inputs.shape
        values = inputs.reshape(self._ch, 1, -1)

        # Add noise or quantize.
        values = quantize(values, self.training, self.quantiles)

        # Evaluate densities.
        lower = self._cdf(values - 0.5, stop_gradient=False)
        upper = self._cdf(values + 0.5, stop_gradient=False)
        with torch.no_grad():
            sign = torch.sign(lower + upper)
            sign[sign == 0] = 1
        likelihood = torch.abs(torch.sigmoid(-sign * upper) - torch.sigmoid(-sign * lower))
        if self._likelihood_bound > 0:
            likelihood = lower_bound(likelihood, self._likelihood_bound)

        # Convert back to input tensor shape.
        values = values.reshape(*shape)
        values = torch.transpose(values, 0, 1)
        likelihood = likelihood.reshape(*shape)
        likelihood = torch.transpose(likelihood, 0, 1)

        return values, likelihood

    def integer_offset_error(self):
        logits = self._cdf(self.quantiles, stop_gradient=True)
        loss = torch.sum(torch.abs(logits - self.target))

        return loss

    def visualize(self, index, minval=-10, maxval=10, interval=0.1):
        # Get the default dtype and device.
        var = next(self.parameters())
        dtype, device = var.dtype, var.device

        # Compute the density.
        x = torch.arange(minval, maxval, interval, dtype=dtype, device=device)
        x_ = torch.zeros(self._ch, 1, len(x), dtype=dtype, device=device, requires_grad=True)
        with torch.no_grad():
            x_[index, 0, :] = x

        w_ = torch.sigmoid(self._cdf(x_, stop_gradient=True))
        w_.backward(torch.ones_like(w_))
        y = x_.grad[index, 0, :]

        # Convert the tensor to numpy array.
        x = x.cpu().numpy()
        y = y.cpu().numpy()

        plt.figure()
        plt.plot(x, y, 'r-')
        plt.show()

        return x, y
