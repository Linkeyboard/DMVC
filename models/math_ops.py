import torch
from torch.autograd import Function


__all__ = [
    "lower_bound",
]


class LowerBoundFunction(Function):
    @staticmethod
    def forward(ctx, inputs, lower_bound):
        outputs = inputs.clamp(min=lower_bound)
        index = (inputs >= lower_bound)
        ctx.save_for_backward(index)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        index = ctx.saved_tensors[0]
        pass_through_if = index | (grad_outputs < 0)
        grad_inputs = pass_through_if.to(grad_outputs.device, grad_outputs.dtype) * grad_outputs

        return grad_inputs, None


# Define the API functions
lower_bound = LowerBoundFunction.apply
