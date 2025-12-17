import infiniccl
import click
import torch
import threading



def test_all_reduce(comm, rank, world_size, device_ids, op):
    infiniccl.set_device(device_type=infiniccl.DEVICE.NVIDIA, device_id=device_ids[rank])
    send_tensor = torch.randn(10, dtype=torch.float32, device='cuda:{}'.format(device_ids[rank]))
    recv_tensor = torch.zeros_like(send_tensor, device='cuda:{}'.format(device_ids[rank]))
    if op == infiniccl.CCLOP.MAX:
        print("local max:", send_tensor.max().item())
    elif op == infiniccl.CCLOP.MIN:
        print("local min:", send_tensor.min().item())
    elif op == infiniccl.CCLOP.SUM:
        print("local sum:", send_tensor.sum().item())
    infiniccl.all_reduce(send_tensor, recv_tensor, op=op, comm=comm)

    if rank == 0:
        if op == infiniccl.CCLOP.MAX:
            print("global max:", recv_tensor.max().item())
        elif op == infiniccl.CCLOP.MIN:
            print("global min:", recv_tensor.min().item())
        elif op == infiniccl.CCLOP.SUM:
            print("global sum:", recv_tensor.sum().item())


def main():
    threads = []
    max_gpus = infiniccl.get_device_count(infiniccl.DEVICE.NVIDIA)
    comms = infiniccl.comm_init_all(device_type=infiniccl.DEVICE.NVIDIA, device_ids=list(range(max_gpus)))
    for op in [infiniccl.CCLOP.MAX, infiniccl.CCLOP.MIN, infiniccl.CCLOP.SUM]:
        print("=" * 60)
        print(f"test all reduce op {op}")
        for rank in range(max_gpus):
            thread = threading.Thread(target=test_all_reduce, args=(comms[rank], rank, max_gpus, list(range(max_gpus)), op))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        
if __name__ == "__main__":
    main()



