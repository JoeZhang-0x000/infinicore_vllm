import torch
import infinicore
import infini_vllm
import pytest

def add_op(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    z = torch.empty_like(x)
    x_infini = infinicore.from_torch(x)
    y_infini = infinicore.from_torch(y)
    z_infini = infinicore.from_torch(z)

    res_infini = x_infini + y_infini
    z_infini.copy_(res_infini)
    return z

@pytest.mark.parametrize("shape", [(1, 2), (3120, 4096)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("device", ["cuda"])
def test_add_op(shape, dtype, device):
    x = torch.randn(shape, dtype=dtype, device=device)
    y = torch.randn_like(x)
    ref = x + y
    res = add_op(x, y)
    assert torch.allclose(res, ref, atol=1e-3, rtol=1e-3)
