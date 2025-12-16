import torch
import infinicore
from typing import List

def rms_norm(
    input: torch.Tensor,
    weight: torch.Tensor,
    normalized_shape: List[int] | torch.Tensor | None = None,
    eps: float = 1e-5,
    *,
    out=None
) -> torch.Tensor:
    if normalized_shape is None:
        normalized_shape = list(weight.shape)
    if isinstance(normalized_shape, torch.Tensor):
        normalized_shape = list(normalized_shape)
    if out is None:
        out = torch.empty_like(input)

    input_infini = infinicore.from_torch(input)
    weight_infini = infinicore.from_torch(weight)
    out_infini = infinicore.from_torch(out)

    res_infini = infinicore.nn.functional.rms_norm(
        input_infini,
        normalized_shape,
        weight_infini,
        eps
    )
    out_infini.copy_(res_infini)
    return out