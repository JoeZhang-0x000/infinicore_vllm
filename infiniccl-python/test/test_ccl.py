import infiniccl
import ctypes
import ray
import torch
import pytest

@ray.remote(num_gpus=1)
class Worker:
    def __init__(self, rank, world_size, raw_unique_id):
        self.rank = rank
        self.world_size = world_size
        
        # 显式设置设备
        infiniccl.set_device(infiniccl.DEVICE.NVIDIA, 0)
        
        # 重建 UniqueID
        self.unique_id = infiniccl.InfinicclUniqueId()
        ctypes.memmove(ctypes.addressof(self.unique_id), raw_unique_id, len(raw_unique_id))
        
        # 初始化通信器指针
        comm_ptr = infiniccl.infinicclComm_t()
        self.comm = infiniccl.InfinicclComm(comm_ptr)
        
        # 初始化分布式环境
        infiniccl.comm_init_rank(
            infiniccl.DEVICE.NVIDIA, 
            self.comm, 
            self.world_size, 
            self.unique_id, 
            self.rank
        )
        print(f"worker {self.rank}/{self.world_size} initialized")

    def test_all_reduce(self, op: infiniccl.CCLOP):
        BUF_SIZE = 1024
        send_buf = torch.randn(BUF_SIZE, dtype=torch.float32, device="cuda:0")
        recv_buf = torch.empty_like(send_buf)

        infiniccl.all_reduce(send_buf, recv_buf, op=op, comm=self.comm)

        return {
            "send_buf": send_buf.cpu(),
            "recv_buf": recv_buf.cpu()
        }

@pytest.fixture(scope="session")
def workers_env():
    num_workers = 8 
    
    ray.init(ignore_reinit_error=True)
    
    unique_id = infiniccl.get_unique_id(infiniccl.DEVICE.NVIDIA)
    raw_unique_id = ctypes.string_at(ctypes.addressof(unique_id), ctypes.sizeof(unique_id))
    
    workers = [Worker.remote(i, num_workers, raw_unique_id) for i in range(num_workers)]
    
    # Ray 启动 Actor 是异步的，这里随便调用一个方法确保初始化完成
    ray.get([w.test_all_reduce.remote(infiniccl.CCLOP.SUM) for w in workers])
    
    yield workers, num_workers
    
    ray.shutdown()

@pytest.mark.parametrize(
    ["infini_op", "torch_op_func"],
    [
        (infiniccl.CCLOP.MAX, lambda x: torch.max(x, dim=0).values),
        (infiniccl.CCLOP.MIN, lambda x: torch.min(x, dim=0).values),
        (infiniccl.CCLOP.SUM, lambda x: torch.sum(x, dim=0)),
    ]
)
def test_all_reduce_ops(workers_env, infini_op, torch_op_func):
    workers, num_workers = workers_env
    
    results = ray.get([w.test_all_reduce.remote(infini_op) for w in workers])

    send_bufs = [res["send_buf"] for res in results]
    recv_bufs = [res["recv_buf"] for res in results]

    expected_buf = torch_op_func(torch.stack(send_bufs))

    for i, recv_buf in enumerate(recv_bufs):
        torch.testing.assert_close(
            recv_buf, 
            expected_buf, 
            rtol=1e-5, 
            atol=1e-5, 
            msg=f"Rank {i} failed for operator {infini_op.name}"
        )
    
    print(f"Operator {infini_op.name} passed verification on all ranks.")
