import torch
import infinicore


def embedding(
    input: torch.Tensor,
    weight: torch.Tensor,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
    *,
    out=None,
) -> torch.Tensor:
    assert (
        (padding_idx is None)
        and (max_norm is None)
        and (scale_grad_by_freq is False)
        and (sparse is False)
    ), "Unsupported parameters."

    assert input.device == "cpu", "Embedding only supports CPU input."

    if out is None:
        out = torch.empty_like(input)
    
    input_infini = infinicore.from_torch(input)
    weight_infini = infinicore.from_torch(weight)
    out_infini = infinicore.from_torch(out)

    res_infini = infinicore.nn.functional.embedding(
        input_infini,
        weight_infini,
        padding_idx=padding_idx,
        max_norm=max_norm,
        norm_type=norm_type,
        scale_grad_by_freq=scale_grad_by_freq,
        sparse=sparse,
        out=None,
    )
    out_infini.copy_(res_infini)
    return out
