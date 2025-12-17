import ctypes
from enum import Enum, unique

# 与 InfiniCore/include/infinicore.h 对应

@unique
class DEVICE(Enum):
    CPU = 0
    NVIDIA = 1
    CAMBRICON = 2
    ASCEND = 3
    METAX = 4
    MOORE = 5
    ILUVATAR = 6
    KUNLUN = 7
    HYGON = 8
    QY = 9

@unique
class DTYPE(Enum):
    INVALID = 0
    BYTE = 1
    BOOL = 2
    I8 = 3
    I16 = 4
    I32 = 5
    I64 = 6
    U8 = 7
    U16 = 8
    U32 = 9
    U64 = 10
    F8 = 11
    F16 = 12
    F32 = 13
    F64 = 14
    C16 = 15
    C32 = 16
    C64 = 17
    C128 = 18
    BF16 = 19

@unique
class STATUS(Enum):
    # Success
    SUCCESS = 0
    # General Errors
    INTERNAL_ERROR = 1
    NOT_IMPLEMENTED = 2
    BAD_PARAM = 3
    NULL_POINTER = 4
    DEVICE_TYPE_NOT_SUPPORTED = 5
    DEVICE_NOT_FOUND = 6
    DEVICE_NOT_INITIALIZED = 7
    DEVICE_ARCHITECTURE_NOT_SUPPORTED = 8
    # Op Errors
    BAD_TENSOR_DTYPE = 10
    BAD_TENSOR_SHAPE = 11
    BAD_TENSOR_STRIDES = 12
    INSUFFICIENT_WORKSPACE = 13

# infiniccl.h
@unique
class CCLOP(Enum):
    SUM = 0
    PROD = 1
    MAX = 2
    MIN = 3
    AVG = 4

infinicclComm_t = ctypes.c_void_p

infinirtStream_t = ctypes.c_void_p

infiniStatus_t = ctypes.c_int
