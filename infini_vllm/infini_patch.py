import infinicore
import torch
import logging
import os
from .utils import print_once


def from_torch(torch_tensor):
    print_once(f"\033[93mUsing Custom from torch\033[0m")
    infini_type = infinicore.utils.to_infinicore_dtype(torch_tensor.dtype)
    infini_device = infinicore.device(torch_tensor.device.type, 0)
    return infinicore.Tensor(
        infinicore.lib._infinicore.strided_from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            list(torch_tensor.stride()),
            dtype=infini_type._underlying,
            device=infini_device._underlying,
        ),
        _torch_ref=torch_tensor,
    )


infinicore.from_torch = from_torch
infinicore.Tensor.from_torch = from_torch
