import torch

try:
    import _genbmm
except ImportError:
    pass


class LogMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, = _genbmm.forward(a, b, 0)
        ctx.save_for_backward(a, b, out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, out = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(a, b, grad_output.contiguous(), out, 0)
        return grad_a, grad_b


class MaxMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, switches = _genbmm.forward(a, b, 1)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(
            a, b, grad_output.contiguous(), switches.float(), 1
        )
        return grad_a, grad_b


class SampleMatMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        out, switches = _genbmm.forward(a, b, 2)
        ctx.save_for_backward(a, b, switches)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        a, b, switches = ctx.saved_tensors
        grad_a, grad_b = _genbmm.backward(
            a, b, grad_output.contiguous(), switches.float(), 2
        )
        return grad_a, grad_b


logbmm = LogMatMul.apply
maxbmm = MaxMatMul.apply
samplebmm = SampleMatMul.apply
