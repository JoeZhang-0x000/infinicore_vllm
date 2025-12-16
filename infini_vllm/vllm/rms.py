from ..functional.rms import rms_norm
from ..utils import print_once
import torch

def rms_forward(self, x, residual = None):
    print_once("\033[91mWarning: rms_forward is not implemented in infini-vllm.\033[0m")
    if self.weight is None:
        self.weight = torch.ones(x.shape[-1], device=x.device, dtype=x.dtype)
    if residual is None:
        return rms_norm(x, self.weight.data, eps=self.variance_epsilon)
    else:
        return rms_norm(x + residual, self.weight.data, eps=self.variance_epsilon), x + residual
