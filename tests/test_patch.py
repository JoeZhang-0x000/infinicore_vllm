import torch
import infinicore
import infini_vllm


a = torch.randn(2, 3, device="cuda")

a_inf = infinicore.tensor.from_torch(a)