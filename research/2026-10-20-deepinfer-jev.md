# 用 DeepInfer 优化 Jev：首轮源码研究

研究日期：2026-10-20。目标是减少决策模型的推理开销，保留原模型的决策头、候选关系与概率计算。

## 已确认的边界

- DeepInfer 当前模型注册表仅注册 `DeepseekV41ForCausalLM`；不是可直接加载任意 Hugging Face 模型的通用后端。[模型注册表](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/model_executor/models/__init__.py#L14-L32)
- 官方 Jev 文档提供托管的 `/v1/systemone` 接口。本次查阅官方文档及公开仓库，未找到可供本地移植的官方 Jev 权重和服务端实现。[官方模型文档](https://docs.typesafe.ai/models)
- 因此，当前能开展源码级优化实验的具体对象是独立项目 **NanoJev / jevinf**，其结果只能归属于这个开放实现。是否能优化官方 Jev 服务内部，仍需要 TypeSafe 的模型与服务端接入条件。
- jevinf 已实现 state → question → candidate 的分段前向与前缀计算复用；不能把已有功能当作本项目贡献。[上游说明](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/README.md)

源码固定于 DeepInfer `8b16149741a4935e1dec8cea27853f273b374c5b`、jevinf `4c9827aeab1e22ca4519158dea2964176d8ba310`。以下机会是源码支持的实验假设，尚未测得加速效果。

## 优先研究的三个入口

| 优先级 | 源码证据 | 可移植的优化思路 | 尚待验证 |
| --- | --- | --- | --- |
| 1 | jevinf 固定选择执行策略，Stage A 与 Stage C 使用行数限制，批内按最长行 padding；上游测量表明最佳策略随输入形状变化。 | 借鉴 DeepInfer 的 token、序列数与上下文预算，根据 state 长度、问题数、候选长度选择策略和分批方式。 | 混合长短输入能否减少 padding、峰值 KV 和 p95 延迟。 |
| 2 | jevinf 的服务路径使用与 benchmark 相同的引擎；阶段计时会同步设备，输出层逐题回传 CPU。 | 分离 profiling 与服务执行，批量处理决策头输出和回传，并研究 CPU 准备与设备执行重叠。 | 保留必要完成同步后，减少中间同步是否产生实际收益。 |
| 3 | jevinf 已复用前缀计算，但分支仍 deepcopy / repeat KV；DeepInfer 用引用计数与分支尾块复制管理共享 block。 | 将只读 state/question 前缀与候选 suffix 的存储分开，研究共享页与写时复制。 | Qwen attention/cache 适配、分支隔离、位置编码、峰值内存及输出一致性。 |

### 1. 按输入形状选择策略和预算

jevinf 的策略与默认分批设置见 [engine.py:76–96](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/engine.py#L76-L96)，padding 见 [engine.py:130–160](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/engine.py#L130-L160)。其[策略测量](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/docs/strategies.md#L52-L70)支持“输入形状影响最佳策略”，并未证明新策略会进一步加速。

DeepInfer 同时检查序列数、批量 token 和上下文总量，见 [scheduler.hpp:440–515](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/scheduler.hpp#L440-L515)。可研究移植预算约束；直接复用该调度器还需要模型、缓存和执行接口适配。

### 2. 减少同步等待

jevinf 的 [device.py:21–28](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/device.py#L21-L28)执行真实的 MPS/CUDA 同步；[Stage C](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/engine.py#L237-L258)在小批次内同步，[输出转换](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/engine.py#L386-L414)逐题处理概率并回传 CPU。[服务入口](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/server.py#L377-L383)调用同一个引擎。

这些同步目前用于阶段计时，不能直接一概删除。应保留 profiling 模式以及端到端计时的完成边界。DeepInfer 的 [worker.py:113–187](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py#L113-L187)提供“先提交当前 forward、再处理上一轮结果”的对照，但其延迟处理限于 decode；NanoJev 的分段 forward 需要单独适配。

### 3. 减少前缀 KV 的物理复制

jevinf 在 [engine.py:164–173](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/engine.py#L164-L173)和 [engine.py:237–258](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/engine.py#L237-L258)复制、扩展 cache；State cache 的拆分也有逐层 clone。

DeepInfer 的 [fork_seq](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/scheduler.hpp#L1085-L1118)和 [block 引用计数](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/csrc/engine/block_manager.hpp#L82-L102)可作为共享存储设计的参考。仅把张量改成 `expand` 不能证明实现了正确的共享缓存；必须检查后续写入、mask、RoPE 位置与释放过程。

## 不能直接套用的优化

- **MTP / DSpark：** NanoJev 的该推理路径使用决策头，并记录 `autoregressive_decode_steps: 0`，没有可通过 draft token 跳过的自回归循环。[engine.py:386–414](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/engine.py#L386-L414)
- **直接开启 DeepInfer ZOS：** 当前代码对 prompt 当场 postprocess，只有 decode 延迟处理；不能声称打开环境变量便能优化 NanoJev。[worker.py:182–187](https://github.com/deepseek-ai/DeepInfer/blob/8b16149741a4935e1dec8cea27853f273b374c5b/deepinfer/worker/worker.py#L182-L187)
- **用字母 token 概率代替原决策头：** 这会改变模型。NanoJev 的 choice head 可跨候选交互，boolean head 也有自己的数学形式；优化必须保留这些关系。[head.py:39–63](https://github.com/zerodegress/jevinf/blob/4c9827aeab1e22ca4519158dea2964176d8ba310/src/jevinf/head.py#L39-L63)

## 实验要求与当前状态

下一阶段先固定 checkpoint、数据与设备，复现上游 baseline，再逐项比较优化前后：原始 logits、概率最大误差、TV/KL、argmax 一致率、端到端 p50/p95、吞吐、峰值内存与 padding token。按 state 长度、问题数和候选数分层报告，包含 warmup，并同步完整设备工作后计时。

**当前完成：源码与官方接口核查。尚未下载模型、移植 DeepInfer 内核或运行性能基准；本项目没有已验证的加速倍数。**
