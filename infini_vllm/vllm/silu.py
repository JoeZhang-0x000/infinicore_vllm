from ..functional.silu import silu as _silu
from ..utils import print_once
from vllm.utils.torch_utils import direct_register_custom_op
import torch


def fake_silu(x):
    return x

direct_register_custom_op("infini_silu", _silu, fake_impl=fake_silu)


def silu_and_mul_forward(self, x):
    print_once("\033[91mWarning: silu_and_mul_forward is not implemented in infini-vllm.\033[0m")
    d = x.shape[-1] // 2
    x_in = x[..., :d]
    y_in = x[..., d:]
    return torch.ops.vllm.infini_silu(x_in) * y_in
