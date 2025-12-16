import torch
import infinicore
import infini_vllm
import pytest

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(1, 32, 1024), (32, 16, 512)])
@pytest.mark.parametrize("device", ["cuda",])
def test_linear(dtype, shape, device):
    M, N, K = shape
    input = torch.randn(M, N, dtype=dtype, device=device)
    weight = torch.randn(K, N, dtype=dtype, device=device)
    bias = torch.randn(K, dtype=dtype, device=device)
    ref = torch.nn.functional.linear(input, weight, bias)
    res = infini_vllm.functional.linear(input, weight, bias)
    assert torch.allclose(res, ref, atol=1e-1)
    