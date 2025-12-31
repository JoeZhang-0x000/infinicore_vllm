# infiniccl-python

`infiniccl-python` 是一个硬件无关的高性能分布式通信库的 Python 封装。它旨在为各种计算后端提供统一的集体通信（Collective Communication）接口，目前已成功适配 NVIDIA (NCCL) 并无缝集成至 vLLM 推理引擎中。

## 依赖
本项目依赖于[InfiniCore](https://github.com/JoeZhang-0x000/InfiniCore), 请先按照指引安装`InfiniCore`。

## 核心特性

- **多后端支持**：底层基于 `InfiniCore`，能够屏蔽不同硬件厂商通信库的差异。
- **高性能绑定**：使用 `ctypes` 直接调用 C++ 接口，支持原始指针传递和 Stream 异步操作。
- **vLLM 无缝集成**：通过 Monkey Patch 技术，无需修改 vLLM 核心源码即可替换底层通信后端。

## 安装

在项目根目录下执行：

```bash
pip install -e .
```

> **注意**：确保系统中已正确安装 `libinfiniccl.so` 和 `libinfinirt.so`（默认路径为 `/root/.infini/lib/`）。

## 与 vLLM 集成使用

只需在你的 vLLM 启动脚本中加入 `import infiniccl`，该库会自动通过 Monkey Patch 接管 vLLM 的 `PyNcclCommunicator`。

```python
import infiniccl  # 必须在初始化 LLM 之前导入
from vllm import LLM

llm = LLM(model="your-model-path", tensor_parallel_size=2)
```

成功集成后，你会在控制台看到如下启动横幅：

```text
*********************************************************
*                                                       *
*   vLLM is now powered by INFINICCL Communication!     *
*                                                       *
*********************************************************
```

## 测试

项目提供了多层次的测试脚本：

### 1. 基础功能测试（单进程模拟）
验证基本的 Python 绑定和算子逻辑：
```bash
python example/test.py
```

### 2. 分布式 Ray 测试
验证在 Ray 分布式环境下的通信正确性：
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
python test/test_ccl.py
```

### 3. 高效 Pytest（推荐）
使用 Pytest Fixture 缓存 Ray 集群，一次性测试所有算子：
```bash
pytest -v -s test/test_ccl.py
```

### 4. vLLM 真实推理测试
验证在实际大模型推理场景下的稳定性：
```bash
python example/test_vllm.py
```

## 当前进度

- [x] 分布式初始化协议 (`commInitRank`)
- [x] AllReduce 算子支持 (SUM, MAX, MIN, AVG)
- [x] vLLM 透明注入 (Monkey Patch)
- [x] Ray/MPI 多进程环境适配
- [ ] AllGather / ReduceScatter 算子支持
- [ ] 完整的异步 Stream 调度优化
- [ ] 不同平台支持
