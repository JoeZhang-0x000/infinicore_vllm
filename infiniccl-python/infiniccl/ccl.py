import ctypes
from infiniccl.infini_enum import *
from infiniccl.torch_utils import *
import torch

ccl_lib_path = "/root/.infini/lib/libinfiniccl.so"
ccl_lib = ctypes.CDLL(ccl_lib_path)

rt_lib_path = "/root/.infini/lib/libinfinirt.so"
rt_lib = ctypes.CDLL(rt_lib_path)

rt_lib.infinirtInit()

INFINICCL_UNIQUE_ID_BYTES = 128
class InfinicclUniqueId(ctypes.Structure):
    _fields_ = [
        ("internal", ctypes.c_byte * INFINICCL_UNIQUE_ID_BYTES),
    ]

ccl_lib.infinicclCommInitAll.argtypes = [
    ctypes.c_int,  # infiniDevice_t
    ctypes.POINTER(infinicclComm_t),  # infinicclComm_t *comms
    ctypes.c_int,  # int ndevice
    ctypes.POINTER(ctypes.c_int),  # const int *device_ids
]
ccl_lib.infinicclCommInitAll.restype = infiniStatus_t

ccl_lib.infinicclGetUniqueId.argtypes = [
    ctypes.c_int,  # infiniDevice_t
    ctypes.POINTER(InfinicclUniqueId),  # InfinicclUniqueId *id
]
ccl_lib.infinicclGetUniqueId.restype = infiniStatus_t

ccl_lib.infinicclCommInitRank.argtypes = [
    ctypes.c_int,  # infiniDevice_t
    ctypes.POINTER(infinicclComm_t),  # infinicclComm_t *comm
    ctypes.c_int,  # int nranks
    InfinicclUniqueId,  # InfinicclUniqueId uniqueId
    ctypes.c_int,  # int rank
]
ccl_lib.infinicclCommInitRank.restype = infiniStatus_t

ccl_lib.infinicclCommDestroy.argtypes = [infinicclComm_t]
ccl_lib.infinicclCommDestroy.restype = infiniStatus_t

ccl_lib.infinicclAllReduce.argtypes = [
    ctypes.c_void_p,  # void *sendbuf
    ctypes.c_void_p,  # void *recvbuf
    ctypes.c_size_t,  # size_t count
    ctypes.c_int,  # infiniDtype_t
    ctypes.c_int,  # infinicclReduceOp_t
    infinicclComm_t,  # infinicclComm_t comm
    infinirtStream_t,  # infinirtStream_t stream
]
ccl_lib.infinicclAllReduce.restype = infiniStatus_t


# 重构接口
class InfinicclError(Exception):
    pass


def check_status(status: int):
    """Check the status code and raise exception if failed"""
    if status != STATUS.SUCCESS.value:
        error_messages = {
            STATUS.INTERNAL_ERROR.value: "Internal error",
            STATUS.NOT_IMPLEMENTED.value: "Not implemented",
            STATUS.BAD_PARAM.value: "Bad parameter",
            STATUS.NULL_POINTER.value: "Null pointer",
            STATUS.DEVICE_TYPE_NOT_SUPPORTED.value: "Device type not supported",
            STATUS.DEVICE_NOT_FOUND.value: "Device not found",
            STATUS.DEVICE_NOT_INITIALIZED.value: "Device not initialized",
            STATUS.DEVICE_ARCHITECTURE_NOT_SUPPORTED.value: "Device architecture not supported",
            STATUS.BAD_TENSOR_DTYPE.value: "Bad tensor dtype",
            STATUS.BAD_TENSOR_SHAPE.value: "Bad tensor shape",
            STATUS.BAD_TENSOR_STRIDES.value: "Bad tensor strides",
            STATUS.INSUFFICIENT_WORKSPACE.value: "Insufficient workspace",
        }
        raise InfinicclError(
            f"Infiniccl error {STATUS(status).name}: {error_messages.get(status, 'Unknown error')}"
        )


class InfinicclComm:
    def __init__(self, comm_ptr: infinicclComm_t):
        self.comm_ptr = comm_ptr

    def __del__(self):
        if self.comm_ptr:
            self.destroy()

    def destroy(self):
        if self.comm_ptr:
            status = ccl_lib.infinicclCommDestroy(self.comm_ptr)
            check_status(status)
            self.comm_ptr = None


def init_runtime():
    status = rt_lib.infinirtInit()
    check_status(status)


def get_device_count(device_type: DEVICE):
    num = ctypes.c_int(0)
    status = rt_lib.infinirtGetDeviceCount(device_type.value, ctypes.byref(num))
    check_status(status)
    return num.value


def set_device(device_type: DEVICE, device_id: int = 0):
    status = rt_lib.infinirtSetDevice(device_type.value, device_id)
    check_status(status)


def create_stream():
    stream = infinirtStream_t()
    status = rt_lib.infinirtStreamCreate(ctypes.byref(stream))
    check_status(status)
    return stream


def destroy_stream(stream: infinirtStream_t):
    status = rt_lib.infinirtStreamDestroy(stream)
    check_status(status)


def sync_stream(stream: infinirtStream_t):
    status = rt_lib.infinirtStreamSynchronize(stream)
    check_status(status)


def device_sync():
    status = rt_lib.infinirtDeviceSynchronize()
    check_status(status)

def get_unique_id(device_type: DEVICE):
    unique_id = InfinicclUniqueId()
    status = ccl_lib.infinicclGetUniqueId(device_type.value, ctypes.byref(unique_id))
    check_status(status)
    return unique_id

def comm_init_rank(
    device_type: DEVICE,
    comm: InfinicclComm,
    nranks: int,
    unique_id: InfinicclUniqueId,
    rank: int,
):
    status = ccl_lib.infinicclCommInitRank(
        device_type.value, ctypes.byref(comm.comm_ptr), nranks, unique_id, rank
    )
    check_status(status)


def comm_init_all(device_type: DEVICE, device_ids: list | None = None):
    if device_ids is None:
        device_count = get_device_count(device_type)
        device_ids = list(range(device_count))

    ndevice = len(device_ids)
    print(f"comm_init_all: device_type={device_type.name}, device_ids={device_ids}")
    device_ids_array = (ctypes.c_int * ndevice)(*device_ids)
    comms_array = (infinicclComm_t * ndevice)()

    status = ccl_lib.infinicclCommInitAll(
        device_type.value, comms_array, ndevice, device_ids_array
    )
    check_status(status)

    return [InfinicclComm(comms_array[i]) for i in range(ndevice)]


def all_reduce(
    send_tensor: torch.Tensor,
    recv_tensor: torch.Tensor,
    op: CCLOP = CCLOP.SUM,
    comm: InfinicclComm = None,
    stream: infinirtStream_t = None,
) -> torch.Tensor:
    if comm is None:
        raise InfinicclError("InfinicclComm is None")

    if stream is None:
        stream = create_stream()
        auto_stream = True
    else:
        auto_stream = False
    send_tensor = send_tensor.contiguous()
    if recv_tensor is not send_tensor:
        recv_tensor = recv_tensor.contiguous()

    count = send_tensor.numel()
    dtype = TORCH_2_INFINI_TYPE.get(send_tensor.dtype, None)
    if dtype is None:
        raise InfinicclError(f"Unsupported dtype {send_tensor.dtype}")
    send_ptr = send_tensor.data_ptr()
    recv_ptr = recv_tensor.data_ptr()

    status = ccl_lib.infinicclAllReduce(
        send_ptr, recv_ptr, count, dtype.value, op.value, comm.comm_ptr, stream
    )
    check_status(status)
    if auto_stream:
        sync_stream(stream)
        destroy_stream(stream)

    return recv_tensor


init_runtime()
