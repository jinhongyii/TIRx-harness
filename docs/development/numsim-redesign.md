# NumSim / Racecheck / Synccheck 重构方案

状态：草案，2026-10-07。分支 `refactor/clean-core`。

本文是重构的工作合约：所有 worker 以此为准，分歧回到这里改文档再改代码。

## 0. 为什么重构

现状（主干 `c3eac62`）：Rust 引擎 192K 行，其中 racecheck 40K、synccheck 25K；Python 源码 10K，Python 测试 103K；frontend-rs 58K。

五个结构性问题，按影响排序：

1. **Checker 是引擎的编译期模式。** `EngineModeImpl`（`engine-rs/src/engine_mode.rs:169-423`）27 个钩子，三个实现，`impl<M> WarpEngine<M>` 7400 行里 153 处 `M::` 分支，`runtime/instructions/mode_axis.rs` 按模式实例化每个指令族，cargo feature 选出三套 rlib，每个 kernel 三个 artifact。钩子里一半是优化策略（`elides_*`、`compacts_*`、`begin_cached_global_read`）而非观察。改一个 checker 要同时推理三个模式。
2. **一个概念实现五到八遍。** mbarrier 的一次 arrive 依次被描述为 ABI 标记类型、plan、引擎方法、hub、`OperationEffect`、strict 重建模型（`strict_*` 共 4.8K 行，作为"正控制"）、`sync_causality` 时钟、`resolved_transition` 日志、`FixedSyncCommandKind`。四到五种 vector clock。racecheck 的共享内存（`race_shadow.rs` 14K）和全局内存（`global_race.rs` 13K）是两套独立引擎。五组 completion-action 类型，十个 `*Hub`。
3. **性能来自未成文的剪枝技巧。** 打包的 `(actor, epoch)` u64 时间戳、单分量 HB 比较、每 range 每 actor 单 witness、exact-hit 原地更新、memo 过的 Arc clock chunk；synccheck 的按资源投影、因果证书、指纹去重、sleep set、COW 协议状态。无 benchmark 守护，重写时丢掉即慢数倍。
4. **构建链把每次实验变成分钟级。** 每个 kernel 一次 rustc O3；引擎 rlib 的 hash 覆盖全部源码，改 checker 一行即全部 artifact 失效；缓存 key 还哈希所有 transpiler `.py`、TVM Python 和 `PATH`。
5. **测试钉实现不钉语义。** 约 110 处断言生成的 Rust 文本，约 160 处断言 poll/transition 计数，约 210 处断言原生 payload 字段；性能门禁是按一台 224 核主机手调的绝对秒数且在默认测试集里；`profile` feature 从未在 CI 打开；**仓库没有测试 CI**（`.github/workflows` 只有 build_wheels、docs、publish_pypi）。

被核实为**不成立**的说法，以免再次误导：

- 生成代码不是 `M` 泛型的，它用具体别名（`artifact_support.rs:222-230`）。去掉 `M` 不会让 per-kernel 编译变快。
- racecheck 发现 race **不会**中止运行，是 `findings.push` 后继续（`race_check.rs:3165` 等）。方案里不需要也不允许"首个 race 即中止"。
- declared word 的"用本次运行观察到的版本当 HB 边"是**退化回退**，不是主路径（`global_race.rs:8779-8800` 注释）。主路径扫描完整写历史取谓词接受的最早一次写，schedule 无关。

## 1. 设计原则

- 一个执行路径，模式是运行期参数。所有模式跑同一份数值/控制/内存路径，分析只观察，不改变程序可见行为（沿用 `numsim/AGENTS.md`）。
- 每个 GPU 概念只有一个转移函数。引擎和 checker 调同一个 `step`。
- 热路径和冷路径分开表示：`Access`（借用、每 kernel 几十万次）和 `SyncEvent`（拥有、几千次）。不做大一统 `Event` 枚举。
- 字节和元数据分开。Arena 只管字节和 validity；分析元数据按 `(alloc, range)` 另放。
- fail closed：预算耗尽、未合并的 shadow、不支持的语义，一律 `incomplete`，不是通过。
- 先量后改。每条剪枝技巧有 criterion 微基准；端到端预算相对于基线文件。

## 2. 目标架构

### 2.1 组件

| 组件 | 职责 | 唯一拥有的数据结构 |
| --- | --- | --- |
| Lowering（Python） | TIRx PrimFunc → `Program` | `Program` |
| Backend: Interpreter | 执行 `Program`，mask 栈重汇聚，循环预算与量子 | `WarpState` |
| Backend: Codegen | `Program` → Rust 文本，每条指令打印为对同一 handler 的调用；rustc 一次 | 生成的 cdylib |
| Scheduler | CTA 为 lockstep 单位，warp 按 id 轮转，seeded；跨 CTA 经 per-CTA inbox 每轮排空一次 | `CtaState`, `Inbox` |
| Arena | 字节 + 每字节 1 bit validity + view/OOB | `Vec<Allocation>` |
| SyncTable | mbarrier / named / cluster / async group / tcgen lifecycle / setmaxnreg 的 `step` 函数 + 待完成队列 | `map<ResourceId, Resource>`, `VecDeque<Completion>` |
| OpLib | 纯数值：cvt、fp8/fp4/bf16、MMA、TMA/im2col/swizzle 地址生成、shfl/reduce。生成 `SUPPORTED_OPS.md` | 静态 op 表 |
| Observer | 约 7 个回调：`access`、`sync`、`async_issue`、`async_complete`、`fence`、`wait_verdicts`、`warp_done`；NumSim 用 no-op | `Vec<SyncEvent>` per warp |
| Racecheck | 在线，FastTrack 风格 shadow；一个 `Clock`，一个 `IntervalShadow` | `map<AllocId, IntervalMap<Cell>>` |
| Synccheck | 离线，对 `SyncEvent` 日志按连通分量投影，sleep-set DFS | `ProtocolTS` |
| Report（Python） | `Finding`、verdict 策略、`to_dict/print/require_clean` | `list[Finding]` |

### 2.2 `Program` 字节码

```rust
pub struct Program {
    pub code: Vec<Instr>,
    pub consts: Vec<Scalar>,
    pub sites: Vec<SiteInfo>,      // 源码位置、dtype、layout 等静态事实
    pub layouts: Vec<TileLayout>,
    pub topology: Launch,
    pub host_abi: Vec<ParamSlot>,
}
```

寄存器是 `[T; 32]` 加 mask。控制流结构化：`If/Else/EndIf` 维护 mask 栈；`LoopIf/LoopEnd` 承担迭代预算、调度量子、自旋停泊三件事。可阻塞指令（`MbarWait`、group wait、`wait_until`）返回 `Blocked(resource)`，调度器下一轮重试，没有 Waker 和等待者注册表。

`Program` 可 serde 序列化。Rust 侧测试直接加载或手写，不需要 Python 和 TVM。

### 2.3 两个后端、一个开关

- `backend = "interp" | "codegen"`，运行期选择，默认先 `interp`。
- Codegen 后端是 `Program` 的打印器：每条 `Instr` 打印成 `handlers::ld_f32(&mut ctx, dst, buf, off, SPACE, SEM, SITE)` 这类调用，寄存器编号、dtype、space 变成常量让 LLVM 内联折叠。**不允许**在生成代码里出现独立的语义实现。
- 两个后端共享 Arena、SyncTable、OpLib、Observer、checker，差别只有分派。性能对比在 corpus 上做，按 kernel 分 NumSim / racecheck / synccheck 三个模式报告。
- 对比结论出来后删掉输的那个。两个后端共存不是最终状态。

### 2.4 内存与 shadow

- 共享/本地/寄存器：CTA 私有，调度线程 `&mut` 直接持有，无锁无 TLS。
- 全局：按 stripe 归属，跨 CTA 可见性靠 inbox 排空处的 fence，不用 per-byte 原子。
- validity 1 bit/byte，NumSim 模式**常开**（当前默认是未初始化读报错）。
- racecheck shadow 按 `(AllocId, range)` 另放，NumSim 模式不分配。

### 2.5 Racecheck 核心

- Cell：`last_write: (actor, epoch, release_clock_ref)`，`reads: Epoch | VC`。
- 一个 `Clock`：actor 集合 = warp ∪ async 虚拟 actor ∪ 跨 CTA actor；proxy（generic/async）是时钟的一个维度。共享内存与全局内存的差别是 actor 集合和 fence 表，不是两个 policy 对象。
- 保留的剪枝技巧，各有 criterion 基准：打包 `(actor, epoch)` u64 时间戳与单分量比较；每 range 每 actor 单 witness，写清掉读；exact-hit 原地更新；Arc 共享 memo 的 clock chunk。
- declared word：引擎在每次 wait 记录**谓词对该字写历史的判定位图**，checker 恢复最早接受的写，保住 schedule 无关的 HB 边。
- 全局 shadow 在 inbox 排空时合并（跨 CTA acquire 到达之处）；launch 结束时未合并的 shadow 是一个合并点，不是默默通过。
- 异步 op 是虚拟 actor，read-side 和 write-side 两个完成里程碑。

### 2.6 Synccheck 核心

- 输入只有 `Vec<SyncEvent>`。
- 按 warp↔resource 连通分量独立探索；状态 = per-warp 游标 + 资源状态 + 待完成多重集；DFS + 状态哈希 + sleep set。
- 保留：按资源投影、因果证书（作为快路径，等 corpus 需要时加）、指纹去重。
- 预算耗尽 → `incomplete` 并报告覆盖。
- 正控制：生产里不再有 strict 副本。保留**一个**几百行的独立参考状态机（无 footprint、无 waiter），仅在 `cfg(test)` 和 nightly 下做 model-based test，随机命令序列驱动真实 `step` 比对。

### 2.7 Python 层

四个模块：`compile.py`（lowering + Program 缓存）、`run.py`（Engine、inputs、bindings）、`report.py`（payload dataclass + 一个渲染器 + JSON schema）、`api.py`（re-export）。一个地方读环境变量。

公开 API 不变：`numsim.transpile`、`Engine().run`、`racecheck()`、`synccheck()`、报告的 `.verdict/.findings/.to_dict/.print/.require_clean`。可见变化只有 `schema_version` 4 → 5。

### 2.8 代码量目标

| 模块 | 行数目标 |
| --- | --- |
| lowering（Python） | ~9K |
| program + interp + scheduler | ~5K |
| arena | ~1.5K |
| sync（step 函数） | ~3K |
| oplib | ~18K（不可压缩，多为搬运） |
| observe | ~0.6K |
| racecheck | ~4K |
| synccheck | ~4K |
| codegen 打印器 | ~2K |
| py | ~1K |
| 合计 | ~48K，对比今天 ~310K |

## 3. 测试策略

允许的 golden 只有三类：op 数值的 GPU 对照、corpus 的位级输出、corpus 的 finding 集合（kind + source anchor + byte overlap）。以快照文件存放，`--update-snapshots` 重生成，CI 不带该 flag 在漂移时失败。

不允许：生成代码文本、poll/transition 计数、payload 内部字段形状、绝对秒数。

层次：
1. 语义一致性：corpus × verdict/findings/输出快照。
2. 差分与属性：解释器 vs codegen 后端位级一致；在线 racecheck vs 回放一致；参考状态机 vs `step`。
3. 纯核心：手写 `Program` 喂 checker，不经 Python。
4. Lowering：断言 `Program` 内容，不断言 Rust 文本。
5. 性能：criterion 微基准 + 端到端相对基线，`performance` marker opt-in，nightly 开 profile。

补测试 CI。

## 4. 迁移：旁路新建，按 kernel 切换

不做原地九步手术。oracle 是 corpus 快照，旧代码一行不改。

0. 冻结 corpus verdict 快照和位级输出；补测试 CI；建 criterion 基准骨架。
1. 新引擎只跑 NumSim（解释器后端），按 corpus kernel 逐个切换，位级对照旧引擎。
2. racecheck 核心，逐 kernel 切换，finding 集合对照。
3. synccheck 与探索器。
4. codegen 后端作为 `Program` 打印器；三模式性能对比；删掉输家。
5. 删旧：`engine-rs/`、`frontend-rs/`、旧 Python 层、钉实现的测试。

### 4.1 并行分工

共享合约先行：`Program`/`Instr`/`Dtype`/`SiteInfo`、`Observer` trait、`Arena` 签名、`SyncTable` 签名、OpLib 函数签名。合约落地后以下并行：

| worker | 范围 | 依赖 |
| --- | --- | --- |
| W1 lowering | TIRx → `Program`，先 vector add 和 microtests 子集，再 corpus | 合约 |
| W2 interp + sched | 解释循环、mask 栈、CTA lockstep、inbox、预算与量子、Arena | 合约 |
| W3 sync | 六个 `step`、Completion、参考状态机 | 合约 |
| W4 oplib | 从 `scalar.rs`/`tcgen_ops.rs`/`tensor_map.rs`/fp-env 搬运数值代码并去掉引擎耦合 | 合约 |
| W5 racecheck | Clock、IntervalShadow、Cell 规则、proxy 维度、declared word 位图 | 合约 + Observer |
| W6 synccheck | SyncEvent 日志、投影、DFS、incomplete 规则 | 合约 + Observer |
| W7 codegen | `Program` 打印器 + 单次 rustc 构建器 | W2 的 handler 接口 |
| W8 python + 测试基建 | compile/run/report/api、快照基建、CI、criterion 骨架 | 合约 |

## 5. 代码位置

- Rust：`tirx_harness/src/tirx_harness/numsim/core-rs/`（workspace：`numsim-core`、`numsim-engine`、`numsim-py`）。
- Python：`tirx_harness/src/tirx_harness/numsim/v2/`，稳定后替换 `numsim/*.py`。
- 旧代码在第 5 步前不动。

## 6. 待决与风险

- 解释器在 scalar 密集 kernel 上可能慢 2 到 5 倍；codegen 后端对比后决定，必要时对直线块加 Cranelift JIT（增量）。
- lowering 重写里 tile form / layout 分析（frontend-rs 约 6K 行）是 op 覆盖回归最先暴露的地方。
- 异步 proxy 的 scoped HB（cluster multicast TMA、部分 tx 完成、cp.async.bulk 只读等待、tcgen05 commit → mbarrier）需要两个完成里程碑，预计要第二轮迭代。
- Synccheck 状态爆炸：16 warp、K 级、N 迭代流水线可能击穿 sleep set，届时加按资源相位计数证书。
- 单 CTA 的 lockstep 意味着大 CTA 小 grid 的 kernel 只用一个核；今天也是如此，不是回归。

## 7. 环境备忘

- 本机 shell 的 `PYTHONPATH`、`TVM_HOME`、`TVM_LIBRARY_PATH`、`LD_LIBRARY_PATH` 指向本地 0.26 的 TVM 开发树，会让 frontend panic（`sym.Analyzer is not registered`）。运行测试前清掉这四个变量，用 `.venv`（`uv sync --locked --extra test`，Python 3.12）。
- 测试从 `tirx_harness/` 目录运行，总是带 `-n`，设 `NUMSIM_WORKER_AFFINITY=off`。
- 引擎改动跑 `cargo test --all-features`。
