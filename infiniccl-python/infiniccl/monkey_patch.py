from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
import infiniccl
import torch

printed_msg = {}
def print_once(msg: str):
    if msg in printed_msg:
        return
    if torch.distributed.get_rank() == 0:
        print(msg)
        printed_msg[msg] = True

global_infiniccl_comm = infiniccl.comm_init_all(device_type=infiniccl.DEVICE.NVIDIA)

def all_reduce(self, input_):
    print_once("\033[92mUsing infiniccl all reduce\033[0m")
    if not hasattr(self, "pynccl_comm"):
        raise ValueError("pynccl_comm must be set")
    rank = self.pynccl_comm.rank

    if not hasattr(self, "infiniccl_comm"):
        if rank == 0:
            print("Initializing infiniccl communicator...")
        self.infiniccl_comm = global_infiniccl_comm
        if rank == 0:
            print("Infiniccl communicator initialized.")
        torch.distributed.barrier()
        if rank == 0:
            print("Barrier passed after init.")

    out = torch.empty_like(input_)
    if rank == 0:
        print(f"Performing all_reduce for rank {rank}...")
    infiniccl.all_reduce(
        send_tensor=input_,
        recv_tensor=out,
        op=infiniccl.CCLOP.SUM,
        comm=self.infiniccl_comm[rank],
    )
    if rank == 0:
        print(f"all_reduce for rank {rank} complete.")
    return out


CudaCommunicator.all_reduce = all_reduce

print("\033[92minfiniccl all_reduce patched\033[0m")