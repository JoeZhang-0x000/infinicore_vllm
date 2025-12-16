import infinicore
import torch


def causal_softmax(input: torch.Tensor, *, out: torch.Tensor | None = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    input_infini = infinicore.from_torch(input)
    out_infini = infinicore.from_torch(out)
    res_infini = infinicore.nn.functional.causal_softmax(
        input_infini
    )
    out_infini.copy_(res_infini)
    return out