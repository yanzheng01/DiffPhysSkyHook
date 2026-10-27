# 方案 B 实施计划：力指令输入通道扩展（面向力输出控制器 / co-design）

> 版本：v1.0（2026-10-27，**待审批**）
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
