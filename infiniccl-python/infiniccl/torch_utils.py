import torch
from infiniccl.infini_enum import DTYPE


TORCH_2_INFINI_TYPE = {
    torch.float32: DTYPE.F32,
    torch.float64: DTYPE.F64,
    torch.int32: DTYPE.I32,
    torch.int64: DTYPE.I64,
    torch.bfloat16: DTYPE.BF16,
    torch.half: DTYPE.F16,
    torch.float16: DTYPE.F16,
}
