import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

import vllm.envs as envs
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream
import infiniccl.ccl as ccl

# TODO: dynamic control
_CURR_DEVICE = ccl.DEVICE.NVIDIA

logger = init_logger(__name__)

_OP_MAP = {
    ReduceOp.SUM: ccl.CCLOP.SUM,
    ReduceOp.PRODUCT: ccl.CCLOP.PROD,
    ReduceOp.MAX: ccl.CCLOP.MAX,
    ReduceOp.MIN: ccl.CCLOP.MIN,
    ReduceOp.AVG: ccl.CCLOP.AVG,
}

class InfiniCommunicator:
    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.NCCL, (
                "PyNcclCommunicator should be attached to a non-NCCL group."
            )
            # note: this rank is the rank in the group
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1 or envs.VLLM_DISABLE_PYNCCL:
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        if self.rank == 0:
            # get the unique id from infiniccl
            self.unique_id = ccl.get_unique_id(_CURR_DEVICE)
            banner = """
*********************************************************
*                                                       *
*   vLLM is now powered by INFINICCL Communication!     *
*                                                       *
*********************************************************
"""
            print(banner)
            logger.info_once(
                "vLLM is using infiniccl", scope="local" 
            )
        else:
            # construct an empty uique id
            self.unique_id = ccl.InfinicclUniqueId()
        
        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)
        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        self.device = device
        self.comm = ccl.InfinicclComm(ccl.infinicclComm_t())
        ccl.comm_init_rank(
            _CURR_DEVICE,
            self.comm,
            self.world_size,
            self.unique_id,
            self.rank
        )

        data = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.all_reduce(data)
        del data


    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor = None,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ) -> torch.Tensor | None:
        if self.disabled:
            return None
        assert in_tensor.device == self.device, (
            f"this nccl communicator is created to work on {self.device}, "
            f"but the input tensor is on {in_tensor.device}"
        )
        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)
        
        # TODO: stream control

        ccl_op = _OP_MAP.get(op, None)
        assert ccl_op is not None, (
            f"ReduceOp {op} is not supported by infiniccl"
        )

        ccl.all_reduce(in_tensor, out_tensor, ccl_op, self.comm)    

    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        raise NotImplementedError()

    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ):
        raise NotImplementedError

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        raise NotImplementedError()

    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        raise NotImplementedError()

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        raise NotImplementedError()

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        raise NotImplementedError()

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        raise NotImplementedError()

    def group_start(self):
        raise NotImplementedError()

    def group_end(self):
        raise NotImplementedError()

    def register_comm_window(self, tensor: torch.Tensor):
        raise NotImplementedError()

    def register_comm_window_raw(self, ptr: int, size: int):
        raise NotImplementedError()

    def deregister_comm_window(self, window):
        raise NotImplementedError()
