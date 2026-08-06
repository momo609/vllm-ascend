# RL Inference Agent 仓库架构设计文档

> AI Agent 驱动的端到端 RL 场景推理适配平台

---

## 目录

- [一、整体架构概览](#一整体架构概览)
- [二、仓库目录结构](#二仓库目录结构)
- [三、核心功能模块设计](#三核心功能模块设计)
  - [3.1 Orchestrator Agent（编排层）](#31-orchestrator-agent编排层)
  - [3.2 Model Adaptation Agent（模型适配 Agent）](#32-model-adaptation-agent模型适配-agent)
  - [3.3 Debug Agent（调试 Agent）](#33-debug-agent调试-agent)
  - [3.4 Precision Agent（精度定位 Agent）](#34-precision-agent精度定位-agent)
  - [3.5 Performance Agent（性能调优 Agent）](#35-performance-agent性能调优-agent)
- [四、AI Agent 驱动机制](#四ai-agent-驱动机制)
  - [4.1 Agent 基类设计](#41-agent-基类设计)
  - [4.2 ReAct 循环](#42-react-循环)
  - [4.3 多 Agent 协作模式](#43-多-agent-协作模式)
- [五、知识库设计](#五知识库设计)
  - [5.1 知识类型](#51-知识类型)
  - [5.2 RAG 增强](#52-rag-增强)
- [六、RL 场景特有设计](#六rl-场景特有设计)
  - [6.1 RL 推理模式适配](#61-rl-推理模式适配)
  - [6.2 RLConnector 设计](#62-rlconnector-设计)
- [七、与现有 vLLM Ascend 项目的集成关系](#七与现有-vllm-ascend-项目的集成关系)
- [八、关键设计决策建议](#八关键设计决策建议)

---

## 一、整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          User Interface Layer                            │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   CLI Chat   │  │   Web UI     │  │   API / SDK  │  │  IDE Plugin  │ │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘ │
└─────────┼──────────────────┼──────────────────┼──────────────────┼───────┘
          │                  │                  │                  │
┌─────────▼──────────────────▼──────────────────▼──────────────────▼───────┐
│                         Agent Orchestration Layer                        │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                       Orchestrator Agent                         │   │
│  │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │   │
│  │   │ Intent   │  │  Task    │  │ Context  │  │ Conversation │   │   │
│  │   │ Parser   │  │ Planner  │  │ Manager  │  │   Memory     │   │   │
│  │   └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────┐
│                         Specialized Agent Layer                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐   │
│  │   Model      │ │   Debug      │ │  Precision   │ │ Performance  │   │
│  │  Adaptation  │ │   Agent      │ │   Agent      │ │    Agent     │   │
│  │    Agent     │ │              │ │              │ │              │   │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘   │
└─────────┼─────────────────┼─────────────────┼─────────────────┼─────────┘
          │                  │                  │                  │
┌─────────▼──────────────────▼──────────────────▼──────────────────▼───────┐
│                           Tool & Skill Layer                             │
│  ┌───────────┐ ┌───────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐  │
│  │  Model    │ │  Debug    │ │ Precision│ │ Profiling │ │ Knowledge │  │
│  │  Tools    │ │  Tools    │ │  Tools   │ │  Tools    │ │   Base    │  │
│  └───────────┘ └───────────┘ └──────────┘ └───────────┘ └───────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────────────┐
│                        Runtime & Infrastructure                          │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐     │
│  │ vLLM     │ │ NPU      │ │ Model    │ │ Benchmark│ │ Monitor  │     │
│  │ Runtime  │ │ Driver   │ │ Converter│ │  Suite   │ │  System  │     │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 二、仓库目录结构

```
rl-inference-agent/
├── pyproject.toml                    # 项目配置与依赖
├── README.md
├── AGENTS.md                         # Agent 开发指南
│
├── agent/                            # Agent 核心层
│   ├── __init__.py
│   ├── orchestrator/                 # 编排 Agent
│   │   ├── __init__.py
│   │   ├── orchestrator.py           # 主编排器：意图解析 → 任务规划 → 调度
│   │   ├── intent_parser.py          # 意图解析器（NL → 结构化意图）
│   │   ├── task_planner.py           # 任务规划器（意图 → 执行计划 DAG）
│   │   ├── context_manager.py        # 上下文管理器（会话状态、中间结果）
│   │   └── memory.py                 # 对话记忆（短期 + 长期 + 经验库）
│   │
│   ├── specialized/                  # 专业化 Agent
│   │   ├── __init__.py
│   │   ├── base.py                   # Agent 基类
│   │   ├── model_adaptation.py       # 模型适配 Agent
│   │   ├── debug_agent.py            # 调试 Agent
│   │   ├── precision_agent.py        # 精度定位 Agent
│   │   └── performance_agent.py      # 性能调优 Agent
│   │
│   └── prompts/                      # Prompt 模板
│       ├── __init__.py
│       ├── system_prompts.py         # 各 Agent 的系统 Prompt
│       └── few_shot_examples.py      # Few-shot 示例库
│
├── tools/                            # 工具层（Agent 可调用的能力）
│   ├── __init__.py
│   ├── registry.py                   # 工具注册中心
│   │
│   ├── model/                        # 模型相关工具
│   │   ├── __init__.py
│   │   ├── converter.py              # 模型格式转换（HF → vLLM / ONNX / MindIR）
│   │   ├── validator.py              # 模型结构校验（算子兼容性、shape 检查）
│   │   ├── quantizer.py              # 量化工具（GPTQ, AWQ, FP8 等）
│   │   └── graph_optimizer.py        # 计算图优化（算子融合、常量折叠）
│   │
│   ├── debug/                        # 调试相关工具
│   │   ├── __init__.py
│   │   ├── layer_dump.py             # 中间层输出 Dump 比对
│   │   ├── nan_detector.py           # NaN/Inf 定位工具
│   │   ├── trace_comparator.py       # Trace 差异对比（CPU vs NPU）
│   │   └── log_analyzer.py           # 日志分析器
│   │
│   ├── precision/                    # 精度相关工具
│   │   ├── __init__.py
│   │   ├── cosine_similarity.py      # 余弦相似度比对
│   │   ├── divergence_locator.py     # 散度定位器（逐层定位精度差异）
│   │   ├── statistical_checker.py    # 统计分布检查（KL 散度、分布偏移）
│   │   └── mixed_precision_advisor.py # 混合精度策略推荐
│   │
│   ├── performance/                  # 性能相关工具
│   │   ├── __init__.py
│   │   ├── profiler.py               # 性能 Profiler（NPU Profiler 封装）
│   │   ├── bottleneck_analyzer.py    # 瓶颈分析器
│   │   ├── memory_analyzer.py        # 显存分析器
│   │   ├── scheduler_tuner.py        # 调度参数调优
│   │   └── benchmark_runner.py       # 基准测试执行器
│   │
│   └── knowledge/                    # 知识库工具
│       ├── __init__.py
│       ├── vector_store.py           # 向量存储（RAG 检索）
│       ├── case_retriever.py         # 历史案例检索
│       └── doc_retriever.py          # 文档检索
│
├── runtime/                          # 运行时适配层
│   ├── __init__.py
│   ├── vllm_adapter.py               # vLLM 运行时适配
│   ├── npu_driver.py                 # NPU 驱动封装
│   ├── distributed.py                # 分布式推理管理
│   └── rl_connector.py              # RL 框架连接器（与训练框架对接）
│
├── knowledge/                        # 知识库（结构化数据）
│   ├── __init__.py
│   ├── model_registry.yaml           # 已知模型注册表
│   ├── precision_patterns.yaml       # 精度问题模式库
│   ├── perf_baselines.yaml           # 性能基线数据
│   └── troubleshooting_guide.yaml    # 故障排查指南
│
├── evaluation/                       # 评估体系
│   ├── __init__.py
│   ├── rl_benchmarks/                # RL 场景 Benchmark
│   │   ├── throughput.py             # 吞吐测试
│   │   ├── latency.py                # 延迟测试
│   │   ├── consistency.py            # 一致性测试（与 reference 比对）
│   │   └── rl_specific.py           # RL 特有场景（beam search, sampling 等）
│   └── metrics.py                    # 指标收集与上报
│
├── cli/                              # 命令行入口
│   ├── __init__.py
│   ├── main.py                       # CLI 主入口
│   └── interactive.py                # 交互式 REPL
│
├── tests/                            # 测试
│   ├── ut/                           # 单元测试
│   ├── e2e/                          # 端到端测试
│   └── fixtures/                     # 测试 Fixtures
│
└── configs/                          # 配置模板
    ├── agent_config.yaml             # Agent 全局配置
    ├── tool_config.yaml              # 工具配置
    └── rl_scenarios/                 # RL 场景预设配置
        ├── ppo.yaml
        ├── grpo.yaml
        └── custom.yaml
```

---

## 三、核心功能模块设计

### 3.1 Orchestrator Agent（编排层）

**职责**：作为系统入口，解析用户意图、制定执行计划、调度专业 Agent。

```
用户输入: "我的 Qwen2.5-7B 模型在 PPO 推理场景下精度不对，帮我定位一下"
                │
                ▼
┌──────────────────────────────┐
│     Intent Parser            │
│  → intent: precision_debug   │
│  → model: Qwen2.5-7B        │
│  → scenario: PPO            │
│  → device: NPU              │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│     Task Planner             │
│  Step 1: 加载模型基线       │
│  Step 2: 运行一致性测试     │
│  Step 3: 逐层精度比对       │
│  Step 4: 定位差异层         │
│  Step 5: 生成修复建议       │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Specialized Agent Dispatch  │
│  → PrecisionAgent.execute()  │
│  → DebugAgent.analyze()      │
│  → ModelAgent.suggest_fix()  │
└──────────────────────────────┘
```

**关键设计点**：

| 组件 | 功能 | 说明 |
|------|------|------|
| Intent Parser | NL → 结构化意图 | 使用 LLM 将自然语言解析为 `{intent, model, scenario, device, params}` |
| Task Planner | 意图 → 执行计划 DAG | 生成有向无环图执行计划，支持并行和条件分支 |
| Context Manager | 会话状态管理 | 维护中间结果，支持多轮交互 |
| Memory | 对话记忆 | 短期记忆（当前会话）+ 长期记忆（跨会话经验）+ 向量化案例库 |

### 3.2 Model Adaptation Agent（模型适配 Agent）

**职责**：负责模型从 HuggingFace 到推理引擎的端到端适配。

**核心流程**：

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 加载模型 │───▶│ 结构分析 │───▶│ 兼容检查 │───▶│ 格式转换 │
└─────────┘    └──────────┘    └──────────┘    └──────────┘
                                                    │
                    ┌───────────────────────────────┘
                    ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ 量化压缩 │───▶│ 图优化   │───▶│ 验证测试 │───▶│ 部署就绪 │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
```

**子能力**：

| 能力 | 工具 | 说明 |
|------|------|------|
| 格式转换 | `ModelConverter` | HF → vLLM / ONNX / MindIR 自动转换 |
| 算子兼容 | `OpValidator` | 扫描不支持的算子，推荐替代方案 |
| 量化 | `Quantizer` | GPTQ / AWQ / FP8 / W8A8 自动量化策略 |
| 图优化 | `GraphOptimizer` | 算子融合、常量折叠、内存规划 |
| RL 适配 | `RLConnector` | 适配 PPO/GRPO 推理模式、KV Cache 策略 |

### 3.3 Debug Agent（调试 Agent）

**职责**：快速定位推理过程中的故障和异常。

**诊断流程**：

```
异常触发（Crash / NaN / OOM）
        │
        ▼
┌───────────────────┐
│  日志收集与分析    │  ← LogAnalyzer（提取关键错误栈）
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  异常分类          │  ← 分类：OOM / 精度 / 算子 / 环境
└────────┬──────────┘
         │
    ┌────┴────┬─────────────┐
    ▼         ▼             ▼
┌───────┐ ┌───────┐   ┌───────────┐
│ OOM   │ │ NaN   │   │ 算子崩溃  │
│ 路径  │ │ 路径  │   │   路径    │
└───┬───┘ └───┬───┘   └─────┬─────┘
    │         │             │
    ▼         ▼             ▼
┌───────────────────────────────────┐
│       根因分析 + 修复建议          │
└───────────────────────────────────┘
```

**核心工具**：

| 工具 | 功能 |
|------|------|
| **LayerDump** | 逐层 dump 中间激活值，与 CPU baseline 逐元素对比 |
| **NaNDetector** | 追踪 NaN/Inf 首次出现的位置（正向 + 反向） |
| **TraceComparator** | 比对两个执行 Trace（如 CPU vs NPU），定位首个差异算子 |
| **LogAnalyzer** | 智能解析日志，提取错误栈、显存分配信息、算子调用序列 |

### 3.4 Precision Agent（精度定位 Agent）

**职责**：RL 场景最关键的模块，负责端到端精度比对与差异定位。

**精度定位流水线**：

```
┌──────────────────────────────────────────────────────────────┐
│                    Phase 1: 端到端一致性检查                   │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐   │
│  │ Reference│    │  Target  │    │ Cosine Similarity    │   │
│  │ (CPU)    │───▶│ (NPU)    │───▶│ KL Divergence         │   │
│  │ Output   │    │  Output  │    │ Top-K Match Rate     │   │
│  └──────────┘    └──────────┘    └──────────┬───────────┘   │
│                                              │               │
│                              不通过 ─────────┘               │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│                  Phase 2: 逐层散度定位                         │
│                                                              │
│   Layer 0 ──► Layer 1 ──► Layer 2 ──► ... ──► Layer N      │
│      │           │           │                  │            │
│      ▼           ▼           ▼                  ▼            │
│   cos=0.999   cos=0.998   cos=0.952 ⚠       cos=0.941 ⚠   │
│                                                              │
│   定位结果: Layer 2 (Attention) 精度开始发散                    │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│               Phase 3: 子算子级精细定位                        │
│                                                              │
│   Layer.Attention:                                           │
│     QKV_Proj: cos=0.999 ✓                                    │
│     RoPE:     cos=0.998 ✓                                    │
│     Softmax:  cos=0.951 ⚠  ← 定位到 Softmax 实现差异          │
│     Output:   cos=0.947 ⚠                                    │
└──────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌──────────────────────────────────────────────────────────────┐
│               Phase 4: 修复建议 & 知识沉淀                     │
│                                                              │
│  建议: Softmax 使用高精度实现 (fp32 中间结果)                   │
│  记录: 存入 knowledge/precision_patterns.yaml                │
└──────────────────────────────────────────────────────────────┘
```

**核心工具**：

| 工具 | 算法 | 说明 |
|------|------|------|
| **DivergenceLocator** | 二分查找 | 逐层定位精度发散位置，在该层内进一步定位到具体算子 |
| **StatisticalChecker** | Cosine Similarity | 余弦相似度（向量方向一致性） |
| **StatisticalChecker** | KL Divergence | KL 散度（概率分布差异） |
| **StatisticalChecker** | Max Relative Error | 最大相对误差（逐元素精度） |
| **StatisticalChecker** | Top-K Match Rate | Top-K 匹配率（对 RL 采样场景尤其重要） |

### 3.5 Performance Agent（性能调优 Agent）

**职责**：面向 RL 推理场景的性能分析与自动调优。

**RL 推理场景特性**：

- 高并发请求（多 environment 并行采样）
- 变长序列（不同 episode 长度不同）
- 混合 decode 模式（prefill + decode 交替）
- KV Cache 复用策略（PPO 多轮采样可复用 prefix）

**调优流水线**：

```
┌──────────────────────────────────────────────────────────────┐
│                    Phase 1: Profiling                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐              │
│  │ NPU      │  │ Memory   │  │ Scheduler    │              │
│  │ Profiler │  │ Profiler │  │  Trace       │              │
│  └────┬─────┘  └────┬─────┘  └──────┬───────┘              │
│       └──────────────┼──────────────┘                       │
│                      ▼                                       │
│              ┌───────────────┐                               │
│              │ Bottleneck    │                               │
│              │  Analyzer     │                               │
│              └───────┬───────┘                               │
└──────────────────────┼──────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  Phase 2: Tuning                             │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Scheduler    │  │ Memory       │  │ Kernel       │      │
│  │ 参数调优     │  │ 配置调优     │  │ 选择调优     │      │
│  │              │  │              │  │              │      │
│  │ - max_batch  │  │ - gpu_mem    │  │ - flash_attn │      │
│  │ - max_seqs   │  │ - block_size │  │ - paged_attn │      │
│  │ - policy     │  │ - swap_space │  │ - fused_ops  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └─────────────────┼─────────────────┘               │
│                           ▼                                   │
│                   ┌───────────────┐                           │
│                   │ Parameter     │                           │
│                   │  Search       │  ← 贝叶斯优化 / 网格搜索  │
│                   └───────────────┘                           │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│               Phase 3: Validation                            │
│                                                              │
│   Throughput  ✓    Latency  ✓    Memory  ✓                  │
│   P50: 12ms       P99: 25ms     Peak: 38GB                   │
└──────────────────────────────────────────────────────────────┘
```

**RL 特有优化维度**：

| 维度 | 策略 | Agent 动作 |
|------|------|-----------|
| Prefix Cache | 复用同一 episode 的多轮采样 prefix | 自动配置 `enable_prefix_caching=True` |
| Batch 策略 | 动态 batch，平衡 prefill/decode 比例 | 自动调整 `max_num_seqs` / `max_num_batched_tokens` |
| Sampling 优化 | 多 temperature / top-p 并行采样 | 推荐 best-of-N 等采样策略参数 |
| KV Cache | RL 场景下 KV Cache 生命周期管理 | 自动配置 block_size 和 swap 策略 |

---

## 四、AI Agent 驱动机制

### 4.1 Agent 基类设计

```python
class BaseAgent:
    """专业化 Agent 基类"""

    # 核心属性
    llm: BaseLLM              # 底层 LLM（用于推理规划）
    tools: dict[str, Tool]    # 可用工具集
    memory: ConversationMemory # 对话记忆
    system_prompt: str        # 系统 Prompt

    # 核心方法
    async def plan(self, task: Task) -> ExecutionPlan: ...
    async def execute(self, plan: ExecutionPlan) -> ActionResult: ...
    async def reflect(self, result: ActionResult) -> Reflection: ...
    async def run(self, user_input: str) -> AgentResponse: ...
```

### 4.2 ReAct 循环

每个专业 Agent 内部采用 **ReAct (Reasoning + Acting)** 模式，作为核心驱动循环：

```
┌─────────────────────────────────────────────┐
│              Agent ReAct Loop                │
│                                             │
│   ┌─────────┐                               │
│   │ THINK   │ ← 分析当前状态，决定下一步     │
│   └────┬────┘                               │
│        │                                    │
│        ▼                                    │
│   ┌─────────┐     ┌──────────────┐         │
│   │  ACT    │────▶│ Tool Calling │          │
│   └────┬────┘     └──────┬───────┘         │
│        │                 │                  │
│        ▼                 ▼                  │
│   ┌─────────┐     ┌──────────────┐         │
│   │ OBSERVE │◀────│ Tool Result  │          │
│   └────┬────┘     └──────────────┘         │
│        │                                    │
│        ▼                                    │
│   ┌─────────┐                               │
│   │ REFLECT │ ← 评估是否达成目标             │
│   └────┬────┘                               │
│        │                                    │
│    完成? ──── No ───▶ 回到 THINK            │
│        │                                    │
│       Yes                                   │
│        │                                    │
│        ▼                                    │
│   ┌─────────┐                               │
│   │ REPORT  │ ← 生成最终报告                 │
│   └─────────┘                               │
└─────────────────────────────────────────────┘
```

**循环步骤说明**：

| 步骤 | 说明 |
|------|------|
| **THINK** | Agent 分析当前上下文和观察结果，推理下一步行动 |
| **ACT** | 根据思考结果，选择合适的 Tool 并调用 |
| **OBSERVE** | 接收 Tool 返回的结果 |
| **REFLECT** | 评估当前进度，判断是否达成目标 |
| **REPORT** | 汇总执行过程，生成结构化报告 |

### 4.3 多 Agent 协作模式

对于复杂任务，Orchestrator 协调多个专业 Agent 协作。以下是一个完整工作流的示例：

```python
# 示例：完整的 RL 模型适配 + 精度定位 + 性能调优
orchestrator.run("""
    帮我适配 Qwen2.5-7B 到 vLLM NPU，用于 PPO 推理场景。
    需要确保精度达标（cosine similarity > 0.99），
    并且吞吐达到 1000 tokens/s 以上。
""")
```

**多 Agent 协作流程**：

```
Step 1: ModelAdaptationAgent
  └─ 模型转换 → 算子兼容检查 → 量化

Step 2: PrecisionAgent
  └─ 端到端精度检查 → 逐层比对 → OK ✓

Step 3: PerformanceAgent
  └─ Profiling → 瓶颈分析 → 参数调优
  └─ 发现吞吐不达标 → 建议使用 FlashAttention + Prefix Cache

Step 4: ModelAdaptationAgent
  └─ 应用建议，重新转换

Step 5: PerformanceAgent
  └─ 重新 Benchmark → 达标 ✓

Step 6: Orchestrator
  └─ 汇总报告，输出最终结果
```

---

## 五、知识库设计

### 5.1 知识类型

知识库为 Agent 提供经验积累和历史案例参考：

```
knowledge/
├── model_registry.yaml           # 模型兼容性矩阵
│   # - model: Qwen2.5-7B
│   #   supported_ops: [RMSNorm, RoPE, FlashAttn, ...]
│   #   known_issues: [Softmax precision on fp16]
│   #   recommended_config: {dtype: bf16, quantization: fp8}
│
├── precision_patterns.yaml       # 精度问题模式
│   # - pattern: "LayerNorm + fp16 overflow"
│   #   symptoms: [cos_sim < 0.95 at LayerNorm output]
│   #   root_cause: "fp16 intermediate overflow"
│   #   fix: "Use fp32 accumulation in LayerNorm"
│   #   affected_models: [Qwen2, Llama3, ...]
│
├── perf_baselines.yaml           # 性能基线
│   # - model: Qwen2.5-7B
│   #   device: Ascend 910B
│   #   baseline:
│   #     throughput: 1500 tokens/s
│   #     latency_p50: 10ms
│   #     memory: 14GB
│
└── troubleshooting_guide.yaml    # 故障排查知识图谱
    # - error: "ACL_ERR_REPEAT_INITIALIZE"
    #   causes: [...]
    #   solutions: [...]
```

**知识库能力矩阵**：

| 知识类型 | 存储格式 | 检索方式 | 更新频率 |
|---------|---------|---------|---------|
| 模型注册表 | YAML | 精确匹配 / 模糊搜索 | 每次新模型接入 |
| 精度模式库 | YAML + 向量 | RAG 检索 | 每次精度问题解决后 |
| 性能基线 | YAML | 精确匹配 | 每次 Benchmark 后 |
| 故障排查指南 | YAML + 向量 | RAG 检索 | 持续积累 |

### 5.2 RAG 增强

Agent 在决策时通过向量检索获取相关知识：

```python
class KnowledgeBase:
    """向量化知识库，支持 RAG 检索"""

    async def retrieve_similar_cases(self, query: str, top_k: int = 5):
        """检索相似的历史案例"""
        # 将当前问题向量化 → 检索最相似的历史案例
        # 返回：案例描述 + 解决方案 + 成功率
        pass

    async def retrieve_precision_patterns(self, symptoms: list[str]):
        """根据症状检索精度问题模式"""
        pass

    async def retrieve_perf_baseline(self, model: str, device: str):
        """检索性能基线数据"""
        pass
```

---

## 六、RL 场景特有设计

### 6.1 RL 推理模式适配

RL 推理与普通 serving 的关键差异：

| 维度 | 普通 Serving | RL 推理 |
|------|-------------|---------|
| 请求模式 | 独立请求 | 批量 episode 采样 |
| 序列关系 | 无关 | 同一 episode 多轮有共同 prefix |
| 延迟要求 | P99 敏感 | 平均延迟敏感（batch 整体） |
| 输出要求 | 确定性 | 采样多样性重要 |
| KV Cache | 用完即弃 | 多轮复用 |

### 6.2 RLConnector 设计

```python
class RLConnector:
    """连接 RL 训练框架与推理引擎"""

    def configure_for_ppo(self, ...):
        """
        配置 PPO 推理模式
        - 启用 prefix caching
        - 配置 sampling 参数 (temperature, top_p, top_k)
        - 设置最优 batch 策略
        """
        pass

    def configure_for_grpo(self, ...):
        """
        配置 GRPO 推理模式
        - 多 group 并行采样
        - advantage 加权采样
        """
        pass

    def monitor_rl_metrics(self):
        """
        RL 特有指标监控
        - 采样多样性 (entropy)
        - KL 散度 (vs reference policy)
        - 有效采样率
        """
        pass
```

---

## 七、与现有 vLLM Ascend 项目的集成关系

```
rl-inference-agent/          ← 新建 Agent 仓库
        │
        │ 调用（通过 runtime/ 适配层）
        ▼
vllm-ascend/                 ← 现有推理后端
    ├── vllm_ascend/device/  ← NPU 算子
    ├── vllm_ascend/worker/  ← 模型运行器
    └── vllm_ascend/patch/   ← 模型 Patch
```

**集成原则**：

- Agent 仓库作为上层编排，通过 `runtime/` 层适配现有推理后端
- **不修改** vllm-ascend 的核心代码
- 通过标准接口（CLI / API / 配置文件）与 vllm-ascend 交互
- 工具层封装 vllm-ascend 的操作，统一为 Agent 可调用的 Tool

---

## 八、关键设计决策建议

| 决策项 | 建议 | 理由 |
|-------|------|------|
| **Agent 框架** | LangGraph | 支持复杂状态机、条件分支、Human-in-the-loop |
| **工具协议** | MCP (Model Context Protocol) | 标准化工具接口，便于工具扩展和跨平台 |
| **向量存储** | Chroma / Milvus | 轻量级向量存储，支持 RAG 检索 |
| **LLM 后端** | 多后端支持（OpenAI / vLLM / 本地） | 灵活适配不同部署环境 |
| **人机协同** | 渐进式自动化 | 初期关键决策由人确认，逐步提升自动化程度 |
| **评估体系** | RL Benchmark Suite | 覆盖精度、性能、稳定性三个维度 |

---

## 附录

### A. Agent 交互示例

**场景：用户报告精度问题**

```
User:   "Qwen2.5-7B 在 PPO 推理时，NPU 输出和 CPU baseline 的 cosine similarity 只有 0.92"

Agent:  [THINK] 这是一个精度问题，需要启动 PrecisionAgent 进行定位。
        [ACT] 调用 KnowledgeBase.retrieve_similar_cases("Qwen2.5-7B PPO 精度")
        [OBSERVE] 找到 2 个相似案例，80% 与 Attention 层实现差异有关

Agent:  [ACT] 启动端到端一致性测试...
        [OBSERVE] cosine_similarity: 0.92, KL_divergence: 0.15
                 Top-1 match rate: 87%, Top-5 match rate: 91%

Agent:  [ACT] DivergenceLocator 逐层扫描...
        [OBSERVE] Layer 14 (Attention) 开始发散: cos=0.93
                 Layer 14.Attention.Softmax: cos=0.88 ⚠
                 Layer 14.Attention.Output: cos=0.90 ⚠

Agent:  [REFLECT] 定位到 Layer 14 的 Softmax 算子是精度差异的根因。
        [ACT] 检索 precision_patterns.yaml...
        [OBSERVE] 匹配模式: "Softmax fp16 precision on Ascend NPU"
                 建议: 使用 fp32 中间结果

Agent:  [REPORT]
        ========================================
        精度定位报告
        ========================================
        模型: Qwen2.5-7B
        场景: PPO 推理
        根因: Layer 14.Attention.Softmax, fp16 精度不足
        建议: 在 softmax 计算中使用 fp32 中间累加
        预期修复后 cos: > 0.99
        参考案例: case-2024-xxx
        ========================================
```

### B. 工具注册示例

```python
from tools.registry import ToolRegistry, tool

@tool(
    name="divergence_locator",
    description="逐层定位模型精度发散位置。输入 reference 和 target 的中间层输出，返回首个发散层索引和精度指标。",
    category="precision"
)
async def divergence_locator(
    reference_outputs: dict[int, torch.Tensor],
    target_outputs: dict[int, torch.Tensor],
    threshold: float = 0.99,
) -> DivergenceResult:
    """二分查找定位精度发散层"""
    ...
```

### C. 场景配置示例

```yaml
# configs/rl_scenarios/ppo.yaml
scenario: ppo
model:
  name: Qwen2.5-7B
  dtype: bf16
  quantization: null   # 可选: fp8, w8a8, gptq

inference:
  max_num_seqs: 256
  max_num_batched_tokens: 32768
  enable_prefix_caching: true  # PPO 关键优化
  block_size: 16

sampling:
  temperature: 1.0
  top_p: 0.9
  top_k: 50

rl_specific:
  num_envs: 128          # 并行 environment 数量
  num_samples_per_prompt: 4  # 每个 prompt 采样数
  advantage_weighted: false  # PPO 不使用 advantage 加权

precision:
  reference_device: cpu
  target_device: npu
  threshold_cosine_similarity: 0.99
  threshold_kl_divergence: 0.01
  threshold_topk_match: 0.95

performance:
  target_throughput: 1000   # tokens/s
  target_latency_p99: 50    # ms
  max_memory: 40            # GB
```