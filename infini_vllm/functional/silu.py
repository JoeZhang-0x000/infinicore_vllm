import torch
import infinicore


def silu(
    input: torch.Tensor, inplace: bool = False, *, out: torch.Tensor | None = None
) -> torch.Tensor:
    input_infini = infinicore.from_torch(input)

    if inplace:
        res_infini = infinicore.nn.functional.silu(input_infini, inplace=True)
        input_infini.copy_(res_infini)
        return input

    if out is None:
        out = torch.empty_like(input)

    out_infini = infinicore.from_torch(out)
    res_infini = infinicore.nn.functional.silu(
        input_infini,
        inplace=False,
    )
    out_infini.copy_(res_infini)
    return out
