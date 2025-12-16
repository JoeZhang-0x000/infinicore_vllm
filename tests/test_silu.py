import torch
import infinicore
import pytest
import infini_vllm


@pytest.mark.parametrize("shape", [(2, 3), (3, 4)])
@pytest.mark.parametrize("dev", ["cuda"])
def test_silu(shape, dev):
    input = torch.rand(shape, dtype=torch.float32, device=dev)
    ans = torch.nn.functional.silu(input)
    out = torch.empty_like(input)
    out = infini_vllm.functional.silu(input, out=out)
    assert torch.allclose(out, ans)
