# 方案 B 实施计划：力指令输入通道扩展（面向力输出控制器 / co-design）

> 版本：v1.1（2026-10-28，**已批准执行并收官**，验收记录见 §9）
> 上游依据：[plan_a_dev_test_plan.md](./plan_a_dev_test_plan.md)（计划 A，已收官，§10 闭环）；[plan_a_skyhook_model.md](./plan_a_skyhook_model.md)（模型规范）
> 背景动机：计划 A 的控制接口是**增益参数**（γ_sky, c_min, c_max），阻尼力 F_d 由 RHS 内部合成（`fd_clip(v₁, v₂)`），不存在外部力输入通道。半主动悬架控制器的直接输出是**阻尼力指令**（尤其 NN 策略 co-design 场景），需要把 F_d 的指令源从"天棚律"替换为"逐步外部指令"，并把 ∂J/∂F_cmd(t)（伴随/影响函数）作为可验证的输出交付——这是后续策略训练的梯度信号接口。

---

## 1. 目标与范围

在计划 A 模型 II（半主动力域 clip）上增加**力指令输入通道 F_cmd(t)**：控制器每步输出期望阻尼力，模型侧 force-domain clip 强制物理可实现性（只耗散、受 c_min/c_max 力窗约束），全时程输出 F_d(t)、clip 激活掩码（指令越界率）、以及整条 ∂J/∂F_cmd 伴随向量。以开环力序列优化冒烟闭环验证后，批量 + GPU 通路就绪，达到 NN 策略 co-design 的接口条件。

**不在本轮范围**：NN 策略与 torch 训练回路本身、stage 级连续指令语义（本方案为逐步 ZOH）、力窗 (c_min, c_max) 与指令的联合优化、作动器动力学（迟滞/速率限制）、真实路面谱。

## 2. 指令语义与接口（关键决策）

1. **指令语义 = 逐步零阶保持（ZOH）**：`f_cmd[n]` 为第 n 步（t_n → t_{n+1}）的力指令，在该步 4 个 RK4 stage 内恒定；记录（F_d、mask）仍按整步口径（细则 4 同源）。不插值。
2. **可实现性由模型侧 clip 强制**：`F_d = clip(f_cmd[n], min(c_min,c_max)·v_rel, max(c_min,c_max)·v_rel)`。任何指令输入都物理可实现化，**优化器侧不做投影**（clip 截断次梯度语义与 D3/D5 一致）；mask 即"指令越界率"诊断，直接可作 co-design 的约束/正则指标。
3. **伴随默认通路 = 反向 Tape**：`Tape(loss=j_val2)` 一次回传得整条 (T,) 梯度 `∂J/∂f_cmd`（与 D5 参数优化默认 JVP 相反——时间序列输入反向一次全得，JVP 每方向一次仅作抽查验证）。反向通路必须 `qd.init(adstack=True)`（计划 A 纪律）。
4. **力窗 (c_min, c_max) 保持为设备参数**，不随指令优化（本扩展范围外，留待 co-design）。
5. **零新数学**：RHS 骨架、J 归约（reduce/finalize）、x_hist 槽位模式、mask 语义全部复用计划 A 已验证机制；唯一新增是指令读取点替换天棚律。

## 3. 阶段计划（E1 → E4，增量铁律沿用计划 A §4：每阶段单变量推进、全绿才进下一阶段）

**E1 · 力指令通道（单轨前向）**
`model2.py`：新增 `f_cmd` 字段 (T_MAX,)、`set_fcmd_elem(n, val)` 写入 kernel、`fd_clip_cmd` 指令版 clip、`rollout_cmd_semi`（骨架同 `rollout_semi`，stage 内 F_d 换 `fd_clip_cmd`）；`sim2.py`：`run_semi_cmd(params, f_cmd, road, x0, ...)` 宿主层（f_cmd 逐元素写入，已知开销、可后优化）。
Gate：T14、T15。**回归门**：计划 A 47 用例保持全绿。

**E2 · 伴随梯度通路**
`f_cmd` 开 needs_grad + needs_dual；反向 Tape 一次得 (T,) 伴随；FwdMode 全长 seed（D4 规则）逐方向抽查；中心差分逐点对照（h = 1e-4，mask 稳定前置断言，确定性采样）。
Gate：T16。

**E3 · 开环力序列优化冒烟**
`opt.py` 扩展 `optimize_force`：对 (n_steps,) 力序列的投影 PGD（无优化器侧投影，见 §2-2）+ Armijo 回退；梯度 = Tape 伴随。起始 f_cmd ≡ 0（等价 c_min 被动阻尼器）。
Gate：T17。

**E4 · 批量 + GPU 通路**
`model2b.py`/`sim2b.py`：`fc_b` 字段 (B_MAX, T_B_MAX) + `rollout_semi_cmd_batch`；`Tape(loss=j_mean)` 一次回传整批 (B, T) 伴随；`gpu_worker` 扩展（先 `ensure_init(arch=qd.cuda)` 再导入、stdout 断言 `arch=cuda`）。
Gate：T18。

## 4. 测试矩阵（T14–T18，接续计划 A 编号）

| # | 阶段 | 测试 | 通过标准 |
|---|---|---|---|
| T14 | E1 | 指令退化锚点：f_cmd ≡ 0 ⟹ F_d = c_min·v_rel（clip 的解析投影），轨迹 ≡ 被动 c_min 阻尼器（vs `run_passive(gamma_s=c_min)`） | RMS rel < 1e-12 |
| T15 | E1 | 任意指令无源性：随机/对抗指令（含大幅越界）下 F_d·v_rel ≥ 0 全时程；mask 统计记录 | 无违例 |
| T16 | E2 | 伴随三路验证：反向 (T,) vs JVP(e_i) vs 中心差分（采样步，mask 稳定前置）；结构断言：n ≥ n_steps 的指令梯度精确 0；kernel J == numpy RMS | rel < 1e-5 |
| T17 | E3 | 开环优化冒烟：J 单调下降段 + 稀疏方向导数 FD 验证（归一化稀疏 -g 子向量，种子确定性重试）+ 可实现性记录 | 单调段 + 方向正确 |
| T18 | E4 | 批量展开一致（指令批量 vs 逐条，全通道）+ CPU–GPU 对拍 + GPU 内三路伴随 | 逐元素 < 1e-12（GPU 三路 rel < 1e-5） |

关键技术风险（T16/T17 共用预案）：力扰动会改变**下游**轨迹，下游 mask 可能翻转使 FD 失效——沿用 D3 锚点：±h 上 mask 位级一致 ⟹ J 在 FD 窗口内光滑；逐点采样 / 稀疏方向 + 确定性种子重试；h 不小于 1e-4（stage 级拐点教训）。

## 5. 交付物

- `skyhook/model2.py` 增量：`f_cmd` 字段、`set_fcmd_elem`、`fd_clip_cmd`、`rollout_cmd_semi`（原 `rollout_semi` 不动）
- `skyhook/sim2.py`：`run_semi_cmd` 宿主层
- `skyhook/model2b.py` / `sim2b.py` 增量：批量指令通道
- `skyhook/opt.py` 增量：`optimize_force`（T 维伴随 PGD）
- `tests/`：`test_e1_force_channel.py`、`test_e2_force_grad.py`、`test_e3_force_opt.py`、`test_e4_force_batch.py`（共享夹具复用 d4_fixture 模式），预计 12–14 个用例
- `tests/gpu_worker.py` 扩展：指令通道 CPU/GPU 对拍
- 本文档 §4 验证记录在执行后闭环回写（同计划 A §10 形式）

## 6. 风险与预案

| 风险 | 预案 |
|---|---|
| 下游 mask 翻转致 FD 失效 | mask 稳定前置断言 + 确定性采样/种子重试；h ≥ 1e-4 |
| f_cmd 逐元素写入的 launch 开销（单轨 1e3、批量 8e3 次/次 setup） | 冒烟规模可接受；如成瓶颈再查 Quadrants 张量拷贝 API，不改语义 |
| T 维反向的内存/耗时 | 单 kernel 反向模式 D2–D4 已验证；必要时缩短冒烟 n_steps |
| 伴随向量被 sqrt(0) 对偶污染 | 计划 A 告诫沿用：批元/工况保证 J > 0（非零 IC 或非零指令） |
| ZOH 语义与未来连续控制器不匹配 | 语义显式落档（§2-1），stage 级指令另立扩展 |
| 侵蚀计划 A 已验证路径 | 原 kernels 不改只增；每阶段跑计划 A 全量回归（47 用例） |

## 7. 工程纪律

全程沿用计划 A §8（英文打印、CUDA 沙箱外、float64、dt = 1 ms 锁定、路面 detach、F_d 显式中间节点、增量单变量）。增补：伴随向量是交付物——E2 起所有测试打印 ∂J/∂f_cmd 的统计摘要（最大/最小/非零元数），供 co-design 评估信号质量。

## 8. 里程碑

- NE1 = E1 通过（指令通道前向就绪，计划 A 回归不回退）
- NE2 = E2 通过（伴随三路验证）
- NE3 = E3 通过（开环优化冒烟）→ 力输出控制器接口语义闭环
- NE4 = E4 通过（批量/GPU）→ co-design（NN 策略）接口条件齐备，本方案收官

---

## 9. 执行结果闭环（E1–E4 验收记录，2026-10-28）

> 本节落档 §5 交付物所要求的验证记录（实测值 vs 通过标准）。E1–E4 全部执行完毕，测试矩阵 T14–T18 全部通过，全量回归 **67 passed（29.93 s，x64 + CUDA，无 skip / xfail）**（计划 A 47 + E1 9 + E2 3 + E3 3 + E4 5），里程碑 **NE1–NE4 全部达成**，方案 B 收官。环境同计划 A §10（quadrants 1.3.0，python 3.12.13，conda env `diffphy`；GPU 用例按纪律沙箱外运行）。共享指令夹具：`tests/d4_fixture.py` 的 `FCMDS`（B=8, N=1000，seed 42 均匀 U(−1, 1) N + b=3 元素 ×8 对抗尖峰，与 D4 的 PARAMS/ROADS/X0S 同源，平路元保持非零 IC）。

### 9.1 各阶段 Gate 验收汇总

| 阶段 | Gate | 通过标准 | 实测 | 判定 |
|---|---|---|---|---|
| E1 | T14/T15 | 见 §4 | T14 指令退化：f_cmd≡0 ⟹ max\|F_d − c_min·v_rel\| = 0（位级）、vs run_passive(γ_s=c_min) rel 3.42e-16；T15 无源性无违例 + fd/mask 与宿主重算位级一致 | ✓ |
| E2 | T16 | 见 §4 | 反向一次得 (T,) 伴随；n ≥ n_steps 梯度精确 0；kernel J == numpy 6.7e-16；FD 12 采样点 rel ≤ 4.5e-09；JVP 全长 seed 3 方向 rel ≤ 1.4e-15；伴随统计 max\|g\| = 5.02e-03、非零 264/1000 | ✓ |
| E3 | T17 | 见 §4 | T17a 零起点平台（zero_gradient 响亮停止，J == 被动锚点，见 §9.5-1）；T17b 冒烟 15/15 迭代严格降 J 1.7718 → 1.7527、越界率 882/1000；T17c 稀疏方向 FD rel ≤ 7.0e-10（起点+终点两处，mask 稳定） | ✓ |
| E4 | T18 | 见 §4 | 批量展开全通道 diff = 0.0（位级）、j vs numpy 2.22e-16；CPU–GPU 对拍最大 8.70e-14（ft），mask 位级 0.0，伴随 1.36e-18；GPU 内三路 rel ≤ 7.36e-08（D4 电池复跑 rel ≤ 1.1e-09） | ✓ |

各阶段测试数：E1 9、E2 3、E3 3、E4 5（CPU 3 + GPU 2），新增合计 **20**，总计 **67**。复验命令（项目根，GPU 用例按纪律沙箱外运行）：`python -m pytest tests/ -q`。

### 9.2 测试矩阵实测明细（T14–T18）

| # | 通过标准 | 实测 | 判定 |
|---|---|---|---|
| T14 | RMS rel < 1e-12 | f_cmd ≡ 0 ⟹ max\|F_d − c_min·v_rel\| = 0.0（逐位），vs run_passive(gamma_s=c_min) RMS rel 3.42e-16；结构记录：零指令在 v_rel ≠ 0 步 mask = 1（指令越界，饱和于窗口边界），v_rel = 0 步窗口零宽（带内 tie） | ✓ |
| T15 | 无违例 | 随机 + 对抗指令（含大幅越界与符号翻转）全时程 F_d·v_rel ≥ 0 无违例（clip 窗口恒与 v_rel 同号，构造性无源）；kernel fd/mask 与宿主重算逐位一致 | ✓ |
| T16 | rel < 1e-5 + 结构断言 | 反向 (T,) vs JVP vs 中心差分（h = 1e-4，mask 稳定前置）：12 个确定性采样点 rel ≤ 4.5e-09，其中 7 个截断步伴随与 FD 同为精确 0；n ≥ n_steps 梯度精确 0；kernel J == numpy RMS 6.7e-16；JVP 全长 seed 3 方向 rel ≤ 1.4e-15；伴随统计（§7 交付物）：max\|g\| = 5.02e-03、非零 264/1000 | ✓ |
| T17 | 单调段 + 方向正确 | T17a：零起点伴随精确全零 ⟹ optimize_force 以 "zero_gradient" 响亮停止且 J == 被动锚点（结构断言锁定）；T17b：非零起点（seed 42 U(−0.2, 0.2)）15/15 迭代严格单调 J 1.7718 → 1.7527，指令越界率 882/1000；T17c：稀疏方向（top-8 归一化 −g）方向导数 FD vs AD rel ≤ 7.0e-10（起点与终点两处，±h mask 位级稳定） | ✓ |
| T18 | 逐元素 < 1e-12（GPU 三路 rel < 1e-5） | 批量 vs 逐条展开（指令批量 vs run_semi_cmd）：state/a2/fd/mask/ft/final_state 全通道 diff = 0.000e+00，j vs numpy RMS 2.220e-16，越界率 80.51%，j_mean = 1.484591860447；批量伴随（一次 Tape 得 (B, T) 整块）：max\|g\| = 7.9998e-04、非零 1600/8000；CPU–GPU 对拍：state 1.13e-14、a2 5.24e-14、force 5.31e-14、mask 0.0（位级）、ft 8.70e-14、final_state 4.89e-15、j 2.67e-15、伴随 1.36e-18；GPU 内三路（带内/混合截断/全饱和三语义槽）：rel AD/FD ≤ 7.36e-08、AD/JVP ≤ 2.60e-15、全饱和槽三路精确 0、激活集稳定全 True | ✓ |

T18 补充（伴随语义三分类，详见 §9.5-2）：带内步 1559/1559 伴随非零；越界记录步 6441 中 41 个伴随非零（RK4 stage 2–4 带内所致，FD 证实 AD 正确）、6400 个全饱和步伴随精确 0（轨迹对 ±h 位级不变 ⟹ FD 精确 0）。批量伴随即 (B, T) 影响函数矩阵，co-design 信号接口就绪。

### 9.3 里程碑达成

- **NE1–NE4 全部达成**：NE1 指令通道前向（E1，计划 A 47 用例不回退）→ NE2 伴随三路验证（E2）→ NE3 开环优化冒烟（E3，力输出控制器接口语义闭环）→ NE4 批量/GPU（E4，co-design 接口条件齐备）。
- 方案 B 就此收官；后续 NN 策略 co-design（torch 训练回路、本模型作为可微仿真器、∂J/∂f_cmd 作为策略梯度信号）另行立项，不在本文档范围。

### 9.4 实际交付物清单（vs §5）

| §5 计划交付物 | 实际落位 | 备注 |
|---|---|---|
| `skyhook/model2.py` 增量 | ✓ | f_cmd 字段 (T_MAX,) needs_grad+needs_dual、set_fcmd_elem、fd_clip_cmd、clip_mask_cmd、rollout_cmd_semi（原 rollout_semi 未动） |
| `skyhook/sim2.py` run_semi_cmd | ✓ | f_cmd 逐元素写入（已知开销，§6 预案口径） |
| `skyhook/model2b.py` / `sim2b.py` 增量 | ✓ | fc_b 字段 (B_MAX, T_B_MAX)、set_fcmd_batch_elem、fd_clip_cmd_b、clip_mask_cmd_b、rollout_semi_cmd_batch；宿主层 _setup_fcmd / run_cmd_j_chain / run_semi_cmd_batch（原批量 kernel 未动） |
| `skyhook/opt.py` optimize_force | ✓ | 无优化器侧投影 PGD + Armijo（alpha₀ = 10），ForceOptResult 含伴随统计 |
| `tests/` 四个测试文件 | ✓ | test_e1_force_channel.py（9）、test_e2_force_grad.py（3）、test_e3_force_opt.py（3）、test_e4_force_batch.py（5，CPU 3 + GPU 2） |
| `tests/gpu_worker.py` 扩展 | ✓ | 指令通道 CPU/GPU 对拍 + 三语义槽三路伴随（D4 电池原样保留） |
| 本文档验证记录 | ✓ | 即本节 §9 |
| （计划外）`tests/d4_fixture.py` 扩展 | ✓ | FCMDS 指令夹具 + perturbed_cmd + fwd_seed_2d（2-D 字段 JVP seed 适配器，§9.5-3） |

### 9.5 执行期关键工程发现（已固化进代码注释与项目记忆）

1. **零指令平台点**（E3/T17a，co-design 关键语义）：f_cmd ≡ 0 ⟹ 伴随**精确全零**——0 恒在力窗 [min(c_min,c_max)·v_rel, max(·)] 之外（v_rel ≠ 0 步饱和于窗口边界、截断次梯度为 0；v_rel = 0 步窗口零宽 tie 亦为 0），且指令仅在本步进入 F_d、无其他通路 ⟹ J 的零指令点是严格平台，梯度优化器以 "zero_gradient" 响亮停止。**开环力序列优化与 NN 策略训练的指令初值必须非零**（T17b 用 U(−0.2, 0.2)）；该性质已作为 T17a 结构断言锁定。
2. **整步 mask ≠ 全 stage 饱和**（E4，修正 E2 措辞的外延）：记录窗口是 stage-1（整步 t_n）状态的窗口，RK4 stage 2–4 中间状态的窗口可以不同——越界**记录**步可有合法非零伴随（夹具上 41/6441，FD 三路证实）。"伴随精确 0" 的准确语义是**全部 4 个 stage 饱和**（此时 F_d 逐 stage 与指令无关，整条轨迹对 ±h 位级不变 ⟹ FD 精确 0）。E2 的 "截断步精确 0" 是全 stage 饱和采样点的特例；f_cmd ≡ 0 全零结论不变（0 对任何 v_rel 都在窗外，§9.5-1）。
3. **FwdMode 2-D seed 适配器**（E4）：FwdMode 的 seed 校验要求 len(seed) == 字段元素总数（flat 断言）且 np.array(seed) 形状 == 字段形状（from_numpy），flat list 无法同时满足 2-D 字段——测试侧以 ndarray 子类桥接（`__bool__ → True` 选"已设置"分支、`__len__ → size` 过 flat 断言、np.array 保形），不加库补丁；实测 AD/JVP ~1e-15（CPU 与 CUDA 一致）。
4. **计划 A 铁律全量复验成立**：单 kernel 反向骨架 + 槽位单写 + `for _ in range(1)` 包裹（E1–E4）、h = 1e-4 激活集稳定（E2/E3/E4 全部 mask 前置断言通过）、sqrt(0) 对偶告诫（批量指令夹具沿用非零 IC）、GPU worker 模式（ensure_init(arch=qd.cuda) 先于 import，stdout 断言 arch=cuda）。

### 9.6 偏差记录（与原文差异及理由）

1. **E3 零起点**：§3 原定"起始 f_cmd ≡ 0"，执行发现其为 J 平台点（§9.5-1），T17b 改用非零随机起点；零起点行为本身转为 T17a 结构断言（响亮停止语义），比原计划更强。
2. **T18 三路槽选取**：由固定槽改为运行时按三类语义槽选取（带内 / 混合截断 / 全饱和）——混合截断步无法先验定位，且语义分类本身就是验证目标；槽位随 npz 落档并打印。
3. **测试粒度**：5 项矩阵细化为 20 个 pytest 用例（T14/T15 各含多断言组，E4 CPU/GPU 分文件分类），通过标准未放松；T18 CPU 侧额外加了批量伴随结构断言（E2 语义批量复验）。
4. **指令夹具量纲**：FCMDS 取 U(−1, 1) N 量级（窗口力尺度 c·v_rel ~ 0.01–1.7 N），实测越界率 80.51%，带内/越界/混合/全饱和四类步齐全，满足全部分域断言。
