---
orphan: true
---

# NumSim / Racecheck / Synccheck 重构方案

状态：实施中，2026-10-08 更新。分支 `refactor/clean-core`。第 0–3 步已完成：v2 在全部 corpus case 上与旧快照一致或有裁定的 delta（`v2-conformance-status.md`）。第 4 步已完成：后端对比结论为保留 `interp`，`codegen` 已删除（`backend-comparison.md`）。第 5 步（删旧）由 `test-migration.md` 的退役计划驱动，尚未执行。

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
- racecheck 发现 race 的处理分两种：全局内存的 race 是 `findings.push` 后继续；共享内存和 TMEM 的 race 在 push 之后返回 `Err` 中止执行（`race_check.rs:3163-3167`，11 处）。新核心统一为继续执行，这是一处行为变化（见 racecheck-semantics.md §8）；"写清掉所有 reader"的规则只在 fail-fast 下成立，新核心只清掉已被该写排序的 reader。
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
| Interpreter（唯一执行器） | 执行 `Program`，mask 栈重汇聚，循环预算与量子 | `WarpState` |
| Scheduler | 驻留 cluster 为分区单位，轮内 warp seeded 轮转；global 经写时复制条带共享，轮末按分区序合并（§2.3） | `Partition`, `CtaState` |
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

### 2.3 后端与调度器（已定）

**后端：只保留解释器。**


- 原计划：`interp` 与 `codegen`（`Program` → Rust 打印器，每条 `Instr` 调用同一 handler，rustc 一次）并存，在 corpus 上按 NumSim / racecheck / synccheck 三模式对比后删掉输家。
- 结论（`backend-comparison.md`）：codegen 在任何 case 上都不比解释器快，核心时间中位数为解释器的 0.87–0.94x，每次调用多约 117 ms，冷构建最长达数小时；解释器比旧引擎快 2.3–3.0x。
- 已执行：`numsim-core/src/codegen/`、`Backend` 枚举、`NUMSIM_V2_BACKEND`、`Engine(backend=...)`、codegen 等价测试与 bench 均已删除。解释器是唯一执行器，每条指令的语义只在 `interp::handlers` 中。

**调度器（`sched/`，分区协议，W6 评审通过，2026-10-08）。**

- 分区：每个驻留 cluster 是一个 `Partition`，自有 CTA、`SyncTable`、`LaunchAux`、事件缓冲与 RNG（种子取 cluster 号）；async op id 按分区编号。cluster 内的跨 CTA 效果（远端 shared 写、远端 mbarrier 命令）同步生效。以下情形整次 launch 用单分区（fallback）：cooperative / `grid.sync`、`max_resident_ctas == 0`、tcgen05 `cta_group::1` 与 `::2` 混用、`RunConfig::single_partition`。分区规则只取决于 program 与 config，与 observer、worker 数无关。
- 一轮三相：
  1. 并行相：各分区在自己的 arena shard 上跑；私有分配原地读写，global/param 经 4 KiB 写时复制条带覆盖轮初状态，故其他 cluster 本轮的 global 写下一轮才可见。worker 数为 1 时也走 shard 路径。
  2. 合并：shard 按分区序合并（同字节后者胜）；首个出错分区之后的分区丢弃。
  3. 串行相（主 arena，分区序）：shard 内的 global RMW（atom/red，以及落到 global 的 bulk/tensor/async 归约）在此逐条执行，不丢更新。
- 事件回放序：分区事件整块回放，序与各分区所见一致——读了别人本轮所写字节的分区先回放，同字节写者保持分区序；分区读回自己本轮所写的字节不算读轮初值。无此序（互读对方所写，SB 型结果）时按分区序回放并报 `incomplete`（`cross_cluster_same_round_cycle`）。`Access::seq` 在回放时分配。
- declared word 历史：launch 级 `WordTable` 为准。分区在副本上记录自己的写；`merge_words` 在回放**之前**按回放序（串行相、`drain_all` 按分区逐个）把新条目按交付序追加，并记录每个本地条目的合并位置，据此改写缓冲中的 `WaitVerdicts`（`observed`、`accepted`）和分区的 verdict 缓存。别的分区本轮的条目对已缓冲的 verdict 记为未接受：等待方的 shard 看不到这些字节，故这样不会多出 HB 边；若无可接受条目，racecheck 报 `WaitExitUnproven`。合并越过 `MAX_WORD_HISTORY` 即 overflow，等待以 `incomplete` 失败关闭。
- 已测不变量：结果与 observer 流与 worker 数无关，有无 observer 结果相同（只有回放序与上面的 cycle 诊断依赖 observer）；verdict 编号等于交付序。测试：
  - `interp_scenarios`：`partitioned_wait_until_is_deterministic_across_workers`（1/8/32 worker）、`partitioned_words_do_not_depend_on_the_observer`、`wait_verdict_indices_follow_the_delivery_order`、`sharded_matches_single_partition_reference`、`moe_synthetic_partitions_and_serial_atomics`；
  - `sched_partition_review`：`every_scenario_is_observer_and_worker_independent`（全部场景 × 3 个种子，1 对 8 worker）、`same_round_writers_each_waiting_on_their_own_value`、`history_overflow_crossed_only_in_the_partition_merge`、`history_overflow_crossed_in_the_serial_phase`；
  - `arena::tests::reading_own_writes_is_not_a_round_start_read`。
- 已取消：CTA 级（cluster 内）分区（peer shared 作写时复制共享状态、远端 mbarrier 按（发送 rank，issue 序）应用、`shared::cluster` 原子作串行点）。HEAD 在主 kernel 上已快于旧引擎：`cudnn_sm100_gemm_proj_rope_mxfp8_bf16in` NumSim，32/8/1 worker 为 4.79 s / 9.31 s / 53.4 s，旧引擎为 12.63 s / 17.80 s / 73.2 s（主机负载 67–131，见 engine-review.md）。

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
- 异步 op 是虚拟 actor，read-side 和 write-side 两个完成里程碑。async group 的完成可见性按 ISA 是**逐线程**的（PTX ISA §9.7.10.28，见 sync-isa-answers.md Q7）：group 按 (warp, lane, domain) 建模，wait 逐 lane acquire，`bulk.wait_group.read` 只加入 read 里程碑。旧实现合并成整 warp 时钟是过于宽松的。
- 共享/TMEM 与全局内存可以统一：15 处差异全部归为 actor 集合、fence/scope 表、表示方式、或需要修正的旧有不一致（racecheck-semantics.md §8），原型用一个 `Checker` 跑通三种空间。

### 2.6 Synccheck 核心

- 输入只有 `Vec<SyncEvent>`。
- 按**资源**投影加上从观测运行取得的 HB gate（不是连通分量：流水线 kernel 的所有 warp 和 barrier 是一个分量，分量投影省不下任何状态；原型在 3000 个随机日志上验证了带 gate 的按资源判定与全程序判定一致）。状态 = per-warp 游标 + 资源状态 + 待完成多重集。
- DFS + 状态哈希 + **strong diamond + 因果证书 + 指纹去重必须与 DFS 一起上线**：16 warp × 4 级 × 32 迭代的流水线上，sleep set 单独仍击穿 10 万状态预算（15 个并发 consumer wait = 每代 2^15），strong diamond 把每代变成链（1062 状态），指纹去重再降到 279，证书降到 9。见 synccheck-explorer.md §3。
- 今天 mbarrier 证书有一个假阳性（"wait 可被越过"在下一代永不完成时也触发，`sync_fixed_unified.rs:1997-2066`）；新实现只在下一代完成时应用。
- clock 和 generation 不需要记录：从一次完整的轮转 schedule 重算，证书再证明对所有 schedule 成立。在线因果追踪器（`sync_causality.rs`）不再需要。
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
| py | ~1K |
| 合计 | ~48K，对比今天 ~310K |

## 3. 测试策略

允许的 golden 只有三类：op 数值的 GPU 对照、corpus 的位级输出、corpus 的 finding 集合（kind + source anchor + byte overlap）。以快照文件存放，`--update-snapshots` 重生成，CI 不带该 flag 在漂移时失败。

不允许：生成代码文本、poll/transition 计数、payload 内部字段形状、绝对秒数。

层次：
1. 语义一致性：corpus × verdict/findings/输出快照。
2. 差分与属性：在线 racecheck vs 回放一致；参考状态机 vs `step`。
3. 纯核心：手写 `Program` 喂 checker，不经 Python。
4. Lowering：断言 `Program` 内容，不断言 Rust 文本。
5. 性能：criterion 微基准 + 端到端相对基线，`performance` marker opt-in，nightly 开 profile。

测试 CI：`.github/workflows/tests.yml`（旧引擎 + `core-rs` 的 `cargo test` + pytest，不含 GPU 与 performance marker）。

## 4. 迁移：旁路新建，按 kernel 切换

不做原地九步手术。oracle 是 corpus 快照，旧代码一行不改。

0. 冻结 corpus verdict 快照和位级输出；补测试 CI；建 criterion 基准骨架。（完成）
1. 新引擎只跑 NumSim（解释器后端），按 corpus kernel 逐个切换，位级对照旧引擎。（完成）
2. racecheck 核心，逐 kernel 切换，finding 集合对照。（完成）
3. synccheck 与探索器。（完成）
4. codegen 后端作为 `Program` 打印器；三模式性能对比；删掉输家。（完成：保留 `interp`，`codegen` 已删除；数据见 `backend-comparison.md`）
5. 删旧：`engine-rs/`、`frontend-rs/`、旧 Python 层、钉实现的测试。（待执行；`scripts/numsim-v2/retire_legacy.py`、`retire_tests.py`）

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
| W7 codegen（已删除） | `Program` 打印器 + 单次 rustc 构建器；对比后删除 | W2 的 handler 接口 |
| W8 python + 测试基建 | compile/run/report/api、快照基建、CI、criterion 骨架 | 合约 |

## 5. 代码位置

- Rust：`tirx_harness/src/tirx_harness/numsim/core-rs/`（workspace：`numsim-types`、`numsim-core`、`numsim-oplib`、`numsim-py`、`numsim-sync-ref`、`numsim-race-core`；crate 划分见 `core-rs/README.md`）。
- Python：`tirx_harness/src/tirx_harness/numsim/v2/`，第 5 步替换 `numsim/*.py`。
- 测试：见 `test-migration.md`「Where tests live」。
- 旧代码在第 5 步前不动。

## 6. 待决与风险

- 解释器性能：codegen 对比已证明打印成 Rust 没有收益（§2.3）；剩余差距在调度器扩展性（25 个大 kernel），必要时再考虑对直线块加 Cranelift JIT（增量）。
- lowering 重写里 tile form / layout 分析（frontend-rs 约 6K 行）是 op 覆盖回归最先暴露的地方。
- 异步 proxy 的 scoped HB（cluster multicast TMA、部分 tx 完成、cp.async.bulk 只读等待、tcgen05 commit → mbarrier）需要两个完成里程碑，预计要第二轮迭代。
- Synccheck 状态爆炸：16 warp、K 级、N 迭代流水线可能击穿 sleep set，届时加按资源相位计数证书。
- 单 CTA 的 lockstep 意味着大 CTA 小 grid 的 kernel 只用一个核；今天也是如此，不是回归。

## 7. 规格文档索引

重构期间产出的规格，都是从旧代码抽取并对照 PTX ISA 核实过的：

- `sync-semantics.md`：六个同步协议的状态、命令、前提、转移、完成、错误；引擎模型与 strict 模型的全部不一致。参考状态机在 `core-rs/numsim-sync-ref/`。
- `sync-isa-answers.md`：七个开放语义问题的 ISA 裁定。四个问题两边模型都错：`tcgen05.alloc` 应阻塞、cluster barrier 要排除已退出线程、elect 后单 lane 进 barrier 是 UB、async group 等待逐线程。
- `sync-behaviour-deltas.md`：相对旧行为的变更清单，供快照 diff 审查。
- `racecheck-semantics.md`：33 条 HB 边、冲突规则、时钟表示技巧与操作数论证、22 条旧有不合理行为。实现在 `numsim-core/src/racecheck/`；原型 `core-rs/numsim-race-core/` 仍在 workspace，但不被任何 crate 依赖。
- `racecheck-isa-answers.md`：release sequence、moral strength、proxy 规则等的 ISA 裁定。
- `synccheck-explorer.md`：两阶段算法、投影、证书、指纹、DFS 剪枝表与测量。实现在 `numsim-core/src/synccheck/`，剪枝基准在 `numsim-core/benches/synccheck.rs`。
- `lowering-inventory.md`：194 个 corpus PrimFunc 的 IR 节点、builtin、dtype、layout、控制流统计；lowering 设计与三个 worked example。

## 8. 环境备忘

- 本机 shell 的 `PYTHONPATH`、`TVM_HOME`、`TVM_LIBRARY_PATH`、`LD_LIBRARY_PATH` 指向本地 0.26 的 TVM 开发树，会让 frontend panic（`sym.Analyzer is not registered`）。运行测试前 `source scripts/dev-env.sh`（清掉这四个变量，设置 `$PY`），环境用 `uv sync --locked --extra test --group benchmark --inexact`（Python 3.12）。详见 `dev-loop.md`。
- 测试从 `tirx_harness/` 目录运行，总是带 `-n`，设 `NUMSIM_WORKER_AFFINITY=off`。
- 旧引擎改动跑 `cargo test --all-features`（`engine-rs`）；新核心跑 `(cd core-rs && cargo test --workspace)`，改 Rust 后用 `core-rs/numsim-py/build_dev.sh` 重建扩展。
