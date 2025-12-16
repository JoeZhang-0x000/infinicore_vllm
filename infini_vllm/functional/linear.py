import infinicore
import torch

def linear(input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty(input.shape[0], weight.shape[0], dtype=input.dtype, device=input.device)
    input_infini = infinicore.from_torch(input)
    weight_infini = infinicore.from_torch(weight)
    if bias is not None:
        bias_infini = infinicore.from_torch(bias)
    out_infini = infinicore.from_torch(out)

    res_infini = infinicore.nn.functional.linear(
        input_infini,
        weight_infini,
        bias_infini if bias is not None else None
    )
    out_infini.copy_(res_infini)
    return out


