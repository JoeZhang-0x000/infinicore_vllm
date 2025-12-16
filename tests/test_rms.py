import torch
import pytest
import infini_vllm

@pytest.mark.parametrize("shape", [(2, 3), (3, 4), (1024, 1024)],)
@pytest.mark.parametrize("device", ["cuda"])
def test_rms_norm(shape, device):
    input = torch.rand(shape, dtype=torch.float32, device=device)
    weight = torch.rand(shape[1], dtype=torch.float32, device=device)
    normalized_shape = weight.shape
    print(normalized_shape)
    ans = torch.nn.functional.rms_norm(input, normalized_shape, weight)
    out = infini_vllm.functional.rms_norm(input, weight)
    assert torch.allclose(out, ans, atol=1e-2)