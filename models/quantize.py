import torch
from torch.autograd import Function


def quantize(x, is_training, offset=0):
    if is_training:
        y = QuantizeFunction.apply(x)
    else:
        y = torch.round(x - offset) + offset

    return y


class QuantizeFunction(Function):
    @staticmethod
    def forward(ctx, inputs):
        noise = torch.zeros(1, dtype=inputs.dtype, device=inputs.device).uniform_(-0.5, 0.5)
        #another method different from directly add the noise
        outputs = torch.round(inputs + noise) - noise

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs
