import infiniccl
import torch
import sys
import vllm.distributed.device_communicators.pynccl as pynccl
from infiniccl.communicator import InfiniCommunicator

pynccl.PyNcclCommunicator = InfiniCommunicator

