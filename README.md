# nano-mooncake

> 纯 Python 实现的 Mooncake 最小原型，理解 KVCache-centric 分离式推理的核心设计

[Mooncake](https://github.com/kvcache-ai/Mooncake) 是 Moonshot AI 开源的分离式 LLM 推理架构，获得 **FAST 2025 最佳论文奖**，在数千节点上每天处理超过 1000 亿 token。原始项目包含 500+ 源文件（C++/Python/Go/Rust），涉及 RDMA、NVLink、CXL 等复杂硬件协议。

**nano-mooncake** 将这套系统的核心设计提炼为纯 Python 实现，覆盖 Transfer Engine、Mooncake Store、vLLM KV Connector 三层完整链路，可直接对接真实 vLLM 运行 Prefill-Decode 分离式推理。

## 架构概览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         nano-mooncake Architecture                       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    vLLM KV Connectors                            │   │
│   │  ┌─────────────────────────┐  ┌────────────────────────────┐    │   │
│   │  │ NanoMooncakeConnector   │  │ NanoMooncakeStoreConnector │    │   │
│   │  │ (P2P: 直接 TE 传输)    │  │ (Store: put/get API)       │    │   │
│   │  └─────────────────────────┘  └────────────────────────────┘    │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    Mooncake Store                                 │   │
│   │  ┌────────────────┐  ┌─────────────┐  ┌───────────────────┐    │   │
│   │  │ MasterService   │  │ StoreClient  │  │ Lease Eviction   │    │   │
│   │  │ (alloc/track)   │  │ (put/get)    │  │ (TTL auto-expire)│    │   │
│   │  └────────────────┘  └─────────────┘  └───────────────────┘    │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │                    Transfer Engine                                │   │
│   │  ┌───────────┐  ┌─────────────┐  ┌──────────────────────────┐  │   │
│   │  │ Segment    │  │ BatchDesc    │  │ TcpTransport             │  │   │
│   │  │ Registry   │  │ (async      │  │ (asyncio, 真实 Mooncake  │  │   │
│   │  │            │  │  tracking)  │  │  用 RDMA 87+ GB/s)       │  │   │
│   │  └───────────┘  └─────────────┘  └──────────────────────────┘  │   │
│   └──────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   ┌──────────────────────────────────────────────────────────────────┐   │
│   │          Metadata Server (HTTP, 跨进程段发现 + KV 定位)         │   │
│   │  ┌──────────────────┐  ┌──────────────────┐                     │   │
│   │  │ MetadataHttpClient│  │ HttpMasterClient  │                     │   │
│   │  │ (段注册/发现)     │  │ (Master RPC 代理) │                     │   │
│   │  └──────────────────┘  └──────────────────┘                     │   │
│   └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

## 文件结构

### 核心实现

| 文件 | 行数 | 说明 | 对应 Mooncake |
|------|------|------|---------------|
| `transfer_engine.py` | 340 | Transfer Engine + TCP Transport + Segment 注册 | `mooncake-transfer-engine/` |
| `mooncake_store.py` | 315 | MasterService + StoreClient + 两阶段 Put + Lease | `mooncake-store/` |
| `metadata_server.py` | 340 | HTTP 元数据服务 + Master API + 跨进程客户端 | `TransferMetadata` (etcd/HTTP) |
| `nano_connector.py` | 423 | vLLM KV Connector（P2P 模式，直接 TE 传输） | `MooncakeConnector` |
| `nano_store_connector.py` | 321 | vLLM KV Connector（Store 模式，put/get API） | `MooncakeStoreConnector` |

### 示例

| 文件 | 行数 | 说明 |
|------|------|------|
| `example_disaggregated.py` | 124 | Prefill/Decode 分离 + TE 传输演示 |
| `example_store.py` | 114 | Mooncake Store put/get 演示 |
| `example_vllm_serving.py` | 385 | vLLM P2P 分离式推理（完整 PD 流水线） |
| `example_store_serving.py` | 379 | vLLM Store 分离式推理（完整 PD 流水线） |

## 核心概念

### 1. Transfer Engine（传输引擎）

Mooncake 最底层的组件。将不同传输协议（TCP/RDMA/NVLink）统一为一套 API：

```python
engine = TransferEngine()
engine.init("node-1", TcpTransport(), metadata, listen_port=9100)
buffer = engine.register_local_memory(size=4*1024*1024)  # 注册 4MB 远程可访问内存

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
- **Batch**：传输请求以 Batch 为单位提交，原子化完成跟踪（`BatchDesc` 计数器）
- **Transport 可插拔**：`Transport` 是抽象基类，`TcpTransport` 用 asyncio 实现 TCP 传输协议（header: `[8B size | 8B offset | 1B opcode | payload]`），真实 Mooncake 还有 `RdmaTransport`（87+ GB/s）和 `NVMeoFTransport`

### 2. Mooncake Store（KVCache 存储）

基于 Transfer Engine 的分布式 KV 存储层，专为 LLM 推理的 KVCache 设计：

```python
master = MasterService(default_lease_seconds=300)
client = StoreClient(master, metadata, "node-1", port=9100)

client.put("req-123:layer-0", kv_data)    # 两阶段 put 协议
data = client.get("req-123:layer-0")       # 查询 Master → TE 远程读取
# 300 秒后 lease 到期，对象自动淘汰
```

**关键设计**：
- **两阶段 Put**：`put_start`（Master 在某个 Segment 上分配 buffer）→ TE 传输数据 → `put_end`（标记完成，启动 lease）
- **Replica 状态机**：`INITIALIZED → PROCESSING → COMPLETE → REMOVED`，防止脏读
- **Lease 淘汰**：对象自带 TTL，过期后由 `run_eviction()` 回收，释放内存

### 3. Metadata Server（元数据服务）

跨进程协调层，将 Mooncake 的 etcd/ZMQ 简化为 HTTP：

```python
# 启动服务器（可选启用 MasterService）
server, store = start_metadata_server(port=8090, master=MasterService())

# 客户端（跨进程透明访问）
meta_client = MetadataHttpClient("http://127.0.0.1:8090")
master_client = HttpMasterClient("http://127.0.0.1:8090")
```

提供两套客户端，均为 duck-typed 设计，可无缝替换进程内对象：

| 客户端 | 替换对象 | API |
|--------|---------|-----|
| `MetadataHttpClient` | `MetadataStore` | `register_segment`, `get_segment`, `save_kv_meta`, `has_kv` |
| `HttpMasterClient` | `MasterService` | `register_segment`, `put_start`, `put_end`, `query` |

### 4. vLLM KV Connectors

两种模式对接 vLLM 的 `KVConnectorBase_V1`，实现 Prefill-Decode 分离式推理：

#### P2P 模式（`NanoMooncakeConnector`）

直接管理 TE buffer，通过 HTTP 记录 KV 位置元数据：

```
Prefill save_kv_layer():
  GPU → CPU → 写入 TE buffer → HTTP 记录 {hash: segment+offset+size}

Decode start_load_kv():
  HTTP 查询位置 → TE READ 远程读 → CPU → GPU 注入 paged KV cache
```

#### Store 模式（`NanoMooncakeStoreConnector`）

通过 StoreClient 委托所有数据管理，Master 集中式分配：

```
Prefill save_kv_layer():
  GPU → CPU → StoreClient.put(key, bytes)
    → Master put_start (分配) → TE 传输 → Master put_end (提交)

Decode start_load_kv():
  StoreClient.get(key)
    → Master query (定位) → TE READ (读取) → CPU → GPU
```

**两种模式的区别**：

| | P2P 模式 | Store 模式 |
|---|----------|-----------|
| 数据管理 | Connector 自己管理 TE buffer | StoreClient 管理 |
| 元数据 | HTTP KV 键值对 | MasterService 集中分配 |
| Buffer 分配 | Connector 内部 `_allocate_buffer()` | Master `put_start()` |
| 对象淘汰 | 无 | Lease TTL 自动淘汰 |
| 适用场景 | 低延迟直传 | 需要集中管理和缓存复用 |

### 5. Prefill/Decode 分离

Mooncake 的核心创新 —— 将 LLM 推理拆分为两个可独立扩缩容的阶段：

| 阶段 | 特点 | 优化目标 |
|------|------|---------|
| **Prefill** | 计算密集，处理整个 prompt | GPU 算力利用率 |
| **Decode** | 内存密集，逐 token 生成 | 内存带宽 / KVCache 访问 |

分离后通过 Transfer Engine 传递 KVCache，Proxy 负责路由：

```
Client → Proxy(:8000) → Prefill(:8010, max_tokens=1) → Decode(:8020, 完整生成)
```

## 运行示例

### 基础示例（无需 vLLM）

```bash
# Prefill/Decode 分离传输演示
python example_disaggregated.py

# Mooncake Store 分布式存取演示
python example_store.py
```

### vLLM 分离式推理（P2P 模式）

自动启动 Metadata Server + Prefill + Decode + Proxy，发送测试请求：

```bash
python example_vllm_serving.py
```

使用 `NanoMooncakeConnector`，数据通过 TE 直接传输。

### vLLM 分离式推理（Store 模式）

```bash
python example_store_serving.py
```

使用 `NanoMooncakeStoreConnector`，数据通过 Mooncake Store 的 put/get API 传输，MasterService 提供集中式分配和 lease 管理。

### 手动部署（独立进程）

```bash
# 1. 启动 Metadata Server（Store 模式加 --enable-master）
python metadata_server.py --port 8090 --enable-master

# 2. 启动 Prefill（P2P 模式）
NANO_METADATA_URL=http://127.0.0.1:8090 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --port 8010 --kv-transfer-config '{
      "kv_connector": "NanoMooncakeConnector",
      "kv_connector_module_path": "nano_connector",
      "kv_role": "kv_both"
    }'

# 2. 启动 Prefill（Store 模式）
NANO_METADATA_URL=http://127.0.0.1:8090 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
    --port 8010 --kv-transfer-config '{
      "kv_connector": "NanoMooncakeStoreConnector",
      "kv_connector_module_path": "nano_store_connector",
      "kv_role": "kv_both"
    }'

# 3. 启动 Decode（同上，换端口 --port 8020）
# 4. 请求发送到 Prefill (max_tokens=1) → Decode (完整生成)
```

## 与真实 Mooncake 的对应关系

### 已实现的核心概念

| nano-mooncake | 真实 Mooncake | 简化内容 |
|---|---|---|
| `TransferEngine` + `TcpTransport` | `mooncake-transfer-engine/` | asyncio TCP 替代 RDMA/NVLink |
| `MetadataStore` + `MetadataHttpClient` | `TransferMetadata` (etcd/Redis/HTTP) | 纯内存 + HTTP |
| `MasterService` + `HttpMasterClient` | `mooncake-store/master/` | 单副本、单锁、bump 分配 |
| `StoreClient` | `mooncake-store/client/` | 无磁盘副本 |
| `Replica` + 状态机 | `Replica`（Memory/Disk/LocalDisk） | 仅内存副本 |
| `BumpAllocator` | OffsetBufferAllocator (bin-based O(1)) | 线性分配、不回收 |
| `NanoMooncakeConnector` | `MooncakeConnector` (P2P) | CPU 中转替代 GPU-Direct |
| `NanoMooncakeStoreConnector` | `MooncakeStoreConnector` | CPU 中转替代 GPU-Direct |
| Disaggregated Proxy | `vllm_v1_proxy_server.py` | stdlib HTTP 实现 |

### 有意简化的生产特性

| 特性 | 真实 Mooncake | nano-mooncake |
|------|-------------|---------------|
| 传输协议 | RDMA 87+ GB/s / NVMeoF / NVLink | TCP (asyncio) |
| Master 并发 | 1024-shard 分片锁 | 单 `threading.Lock` |
| 副本策略 | 多副本 + 条带化 + 热块复制 | 固定 1 副本 |
| 分配算法 | FreeRatioFirst / Random / CXL | 顺序 first-fit |
| 淘汰策略 | 近似 LRU + 水位线 + Soft Pinning | 仅 Lease TTL |
| Master HA | etcd leader 选举 | 单进程 |
| GPU 传输 | GPUDirect RDMA / cuFile | GPU → CPU → TE → CPU → GPU |
| Multi-NIC | 拓扑感知 + slice 并行 + 带宽聚合 | 单连接 |

### 未涉及的模块

| 模块 | 说明 |
|------|------|
| P2P Store | BitTorrent 式去中心化对象共享（checkpoint 分发） |
| 多层存储 | 内存 → SSD → DFS 分层，透明降级读取 |
| SGLang HiCache | 三级缓存 (GPU/Host/Store) + 预取 |
| Conductor | 全局调度器，热度预测 + 冷热迁移 |
| Expert Parallelism | MoE 专家并行 (CUDA) |

## 数据流详解

### P2P 模式数据流

```
                    ┌───────────────────┐
                    │  Metadata Server  │
                    │  (:8090)          │
                    │  - 段注册/发现     │
                    │  - KV 位置存储    │
                    └───┬─────────┬─────┘
                 register     lookup
                 /save_kv     /load_kv
                    ┌───┴───┐ ┌───┴───┐
                    │Prefill│ │Decode │
                    │(:8010)│ │(:8020)│
                    │       │ │       │
                    │ TE buf│ │ TE buf│
                    └───┬───┘ └───┬───┘
                        │  TCP TE │
                        └─────────┘
                     直接读写 TE buffer
```

1. Prefill `save_kv_layer()`: GPU → CPU → 写入本地 TE buffer → HTTP 记录位置
2. Decode `start_load_kv()`: HTTP 查询位置 → TE READ 远程读取 → CPU → GPU

### Store 模式数据流

```
                    ┌───────────────────┐
                    │  Metadata Server  │
                    │  (:8090)          │
                    │  - 段注册/发现     │
                    │  - MasterService  │
                    │    (分配/跟踪/淘汰)│
                    └───┬─────────┬─────┘
              put_start/end  query/get
              register_seg   discover
                    ┌───┴───┐ ┌───┴───┐
                    │Prefill│ │Decode │
                    │Store  │ │Store  │
                    │Client │ │Client │
                    │(:8010)│ │(:8020)│
                    └───┬───┘ └───┬───┘
                        │  TCP TE │
                        └─────────┘
                  Store 管理的 TE 传输
```

1. Prefill `save_kv_layer()`: GPU → CPU → `StoreClient.put()` → Master 分配 → TE 传输 → Master 提交
2. Decode `start_load_kv()`: `StoreClient.get()` → Master 查询副本位置 → TE READ → CPU → GPU

### 两阶段 Put 协议

```
StoreClient                    MasterService                   Remote Segment
    │                              │                                │
    │── put_start(key, size) ─────>│                                │
    │                              │── allocate on segment ────────>│
    │<── Replica{segment, offset} ─│                                │
    │                              │                                │
    │── TE WRITE ─────────────────────────────────────────────────>│
    │<── transfer complete ───────────────────────────────────────  │
    │                              │                                │
    │── put_end(key) ─────────────>│                                │
    │                              │── mark COMPLETE, start lease   │
    │<── ok ───────────────────────│                                │
```

## 学习路径建议

1. **`transfer_engine.py`**：理解 Segment 注册、Batch 传输、TCP 线缆协议
2. **`mooncake_store.py`**：理解两阶段 Put、Replica 状态机、Lease 淘汰
3. **`example_disaggregated.py`**：运行，体验 Prefill/Decode 分离的数据流
4. **`example_store.py`**：运行，体验 Store 的 put/get 和 lease 淘汰
5. **`metadata_server.py`**：理解跨进程协调的 HTTP 化
6. **`nano_connector.py`**：理解 vLLM KV Connector 如何桥接 GPU 和 TE
7. **`nano_store_connector.py`**：对比 P2P 模式，理解 Store 模式的区别
8. **`example_vllm_serving.py`** / **`example_store_serving.py`**：运行完整 PD 推理
9. **对照 [Mooncake 源码](https://github.com/kvcache-ai/Mooncake)**：看真实实现如何扩展这些核心概念

## 参考资料

- [Mooncake 源码](https://github.com/kvcache-ai/Mooncake) — 完整的生产级实现
- [Mooncake 文档](https://kvcache-ai.github.io/Mooncake/) — Transfer Engine、Store、vLLM 集成
- [FAST 2025 论文](https://www.usenix.org/conference/fast25/presentation/qin) — Mooncake: Kimi's KVCache-centric Architecture for LLM Serving
- [vLLM KV Connector 文档](https://docs.vllm.ai/en/latest/design/v1/kv_connector_v1.html) — V1 Connector 接口规范
