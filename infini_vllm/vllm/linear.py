from ..functional.linear import linear
from ..utils import print_once


def infini_unquantized_gemm(
    layer,
    x,
    weight,
    bias
):
    return linear(x, weight, bias)

def dispatch_unquantized_gemm():
    print_once("\033[32mInfini GEMM is enabled.\033[0m")
    return infini_unquantized_gemm