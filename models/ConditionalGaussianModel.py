import math

import torch
import torch.nn as nn

from .math_ops import lower_bound


class ConditionalGaussianModel(nn.Module):
    def __init__(self, scale_bound, likelihood_bound):
        super().__init__()

        self._scale_bound = scale_bound
        self._likelihood_bound = likelihood_bound

        self.register_buffer("_const", torch.tensor(-(2 ** -0.5)))
        self.register_buffer("_half", torch.tensor(1/2))

    def _standardized_cumulative(self, inputs):
        return self._half * torch.erfc(self._const * inputs)

    def forward(self, inputs, loc, scale_minus_one):
        scale = lower_bound(scale_minus_one + 1, self._scale_bound)
        values = torch.abs(inputs - loc)
        upper = self._standardized_cumulative((.5 - values) / scale)
        lower = self._standardized_cumulative((-.5 - values) / scale)
        likelihood = upper - lower
        likelihood = lower_bound(likelihood, self._likelihood_bound)

        return likelihood
