# nano-mooncake

> ~500 行 Python 代码理解 Mooncake 的核心设计

[Mooncake](https://github.com/kvcache-ai/Mooncake) 是 Moonshot AI 开源的 **KVCache-centric 分离式 LLM 推理架构**，获得 FAST 2025 最佳论文奖。原始项目包含 500+ 源文件（C++/Python/Go/Rust），涉及 RDMA、NVLink 等复杂硬件协议。

**nano-mooncake** 用纯 Python 提取了 Mooncake 的核心设计思想，帮助你快速理解这个系统的精髓。

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                    nano-mooncake Architecture                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐                      ┌──────────────┐         │
│  │ Prefill Node  │   KVCache Transfer  │ Decode Node   │         │
│  │ (compute KV)  │ ==================> │ (generate)    │         │
│  └───────┬──────┘                      └───────┬──────┘         │
│          │                                      │                │
│  ┌───────┴──────────────────────────────────────┴──────┐        │
│  │              Transfer Engine (TCP)                    │        │
│  │  ┌─────────┐  ┌───────────┐  ┌────────────────┐    │        │
│  │  │ Segment  │  │ BatchDesc  │  │ TransferRequest│    │        │
│  │  │ Registry │  │ (tracking) │  │ (READ/WRITE)   │    │        │
│  │  └─────────┘  └───────────┘  └────────────────┘    │        │
│  └─────────────────────┬───────────────────────────────┘        │
│                        │                                         │
│  ┌─────────────────────┴───────────────────────────────┐        │
│  │              Mooncake Store                          │        │
│  │  ┌────────────────┐  ┌─────────┐  ┌───────────┐    │        │
│  │  │ MasterService   │  │ Replica  │  │ Lease TTL │    │        │
│  │  │ (put/get/query) │  │ tracking │  │ eviction  │    │        │
│  │  └────────────────┘  └─────────┘  └───────────┘    │        │
│  └──────────────────────────────────────────────────────┘        │
│                        │                                         │
│  ┌─────────────────────┴───────────────────────────────┐        │
│  │           Metadata Server (HTTP/in-memory)           │        │
│  └──────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## 文件结构

| 文件 | 行数 | 说明 |
|---|---|---|
| `transfer_engine.py` | ~200 | Transfer Engine + TCP Transport |
| `mooncake_store.py` | ~200 | 分布式 KVCache Store（Master/Client） |
| `metadata_server.py` | ~80 | HTTP 元数据服务器 |
| `example_disaggregated.py` | ~100 | 演示 Prefill/Decode 分离 |
| `example_store.py` | ~90 | 演示 KVCache Store put/get |

## 核心概念

### 1. Transfer Engine（传输引擎）

Mooncake 最核心的组件。将不同传输协议（TCP/RDMA/NVLink）统一为一套 API：

```python
engine = TransferEngine()
engine.init("node-1", TcpTransport(), metadata, listen_port=9100)
buffer = engine.register_local_memory(size=4*1024*1024)  # 注册 4MB 本地内存

# 批量异步传输
batch_id = engine.allocate_batch_id(batch_size=1)
engine.submit_transfer(batch_id, [
    TransferRequest(opcode=WRITE, source_offset=0,
                    target_id="node-2", target_offset=0, length=1024)
])
status = engine.wait_for_completion(batch_id)
```

**关键设计**：
- **Segment**：节点将本地内存注册为 Segment，其他节点通过名字查找并远程读写
- **Batch**：传输请求以 Batch 为单位提交，支持原子化的完成跟踪
- **Transport 可插拔**：TCP 只是最简单的实现，真实 Mooncake 支持 RDMA（87 GB/s）等

### 2. Mooncake Store（KVCache 存储）

基于 Transfer Engine 的分布式 KV 存储，专为 LLM 推理的 KVCache 设计：

```python
master = MasterService(default_lease_seconds=300)
client = StoreClient(master, metadata, "node-1", port=9100)

client.put("request-123:layer-0", kv_data)    # 存储 KVCache
data = client.get("request-123:layer-0")       # 读取 KVCache
# 300 秒后自动过期淘汰
```

**关键设计**：
- **两阶段 Put**：`put_start`（Master 分配 buffer）→ 数据传输 → `put_end`（标记完成）
- **Replica**：数据副本，状态机 `INITIALIZED → PROCESSING → COMPLETE → REMOVED`
- **Lease 机制**：对象自动过期，释放宝贵的内存资源

### 3. Prefill/Decode 分离

Mooncake 的核心创新 —— 将 LLM 推理拆分为两个阶段：

| 阶段 | 特点 | 优化目标 |
|---|---|---|
| **Prefill** | 计算密集，处理整个 prompt | GPU 算力利用率 |
| **Decode** | 内存密集，逐 token 生成 | 内存带宽 / KVCache 访问 |

分离后通过 Transfer Engine 传递 KVCache，让每个阶段都能独立扩缩容。

## 运行示例

```bash
# 示例 1: Prefill/Decode 分离推理
python example_disaggregated.py

# 示例 2: KVCache Store 分布式存储
python example_store.py

# 独立运行 Metadata Server
python metadata_server.py --port 8080
```

## 与原始 Mooncake 的对应关系

| nano-mooncake | 原始 Mooncake | 简化内容 |
|---|---|---|
| `TransferEngine` | `mooncake-transfer-engine/` | 只保留核心 API |
| `TcpTransport` | `tcp_transport.cc` (1500行 C++) | asyncio 实现 |
| `MetadataStore` | `TransferMetadata` (etcd/HTTP) | 内存字典 |
| `MasterService` | `mooncake-store/master/` | 单副本、单锁 |
| `StoreClient` | `mooncake-store/client/` | 无磁盘副本 |
| `Replica` | `Replica`（3种类型） | 仅内存副本 |
| `BumpAllocator` | CacheLib slab allocator | 线性分配 |
| — | RDMA/NVLink/CXL Transport | 省略（只保留 TCP） |
| — | 拓扑感知路由 | 省略 |
| — | 多副本复制 + 条带化 | 省略 |
| — | P2P Store (Go) | 省略 |
| — | Expert Parallelism (CUDA) | 省略 |

## 学习路径建议

1. **先读 `transfer_engine.py`**：理解 Segment 注册、Batch 传输、TCP 协议
2. **再读 `mooncake_store.py`**：理解两阶段 Put、Replica 状态机、Lease 淘汰
3. **运行 `example_disaggregated.py`**：体验 Prefill/Decode 分离的数据流
4. **运行 `example_store.py`**：体验 KVCache 的分布式存取
5. **对照原始 Mooncake 源码**：看真实实现如何扩展这些核心概念
