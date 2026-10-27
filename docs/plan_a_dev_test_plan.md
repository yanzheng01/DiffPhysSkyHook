# 方案 A 实施计划：可微分 Skyhook 动力学模型开发与测试（Quadrants + RK4）

> 版本：v1.2（2026-10-27，已批准；同日 D0–D5 执行完毕，§10 闭环收档）
> 上游依据：[plan_a_skyhook_model.md](./plan_a_skyhook_model.md)（模型规范）；[skyhook_model_research.md](./skyhook_model_research.md)（调研结论）
> 决策记录：① 框架 = **Quadrants**（Genesis 编译器/内核层，**单轨：仅编译后端**）；② 积分器 = **RK4**（dt = 1 ms），半隐式 Euler 仅作收敛阶对照；③ ON-OFF 天棚基线**纳入**（仅前向仿真，不参与梯度）；④ **多实现对拍已裁撤**（2026-10-27），数学正确性由解析锚点 + 差分梯度兜底。

---

## 1. 目标与范围

开发模型 I（理想天棚，线性解析基准）与模型 II（半主动力域 clip，可微优化主体）的可微分动力学实现，含 ON-OFF 对照分支，全部通过 13 项验收测试后进入参数优化与 co-design 阶段。

**不在本轮范围**：整车多体扩展、轮胎-路面接触、NN 策略 co-design、真实路面谱标定。

## 2. 单轨实现架构（Quadrants 编译后端）

模型唯一实现 = Quadrants 编译后端：RHS 以 `@qd.func`/`@qd.kernel` 编写，编译到本机后端执行；CUDA 冒烟按纪律沙箱外运行。Python backend 仅在开发调试时临时可用，不入验收体系。

**架构约定**：
- 时间步进循环留在 **Python 宿主层**，每步调用 kernel；RHS 内不写运行时循环，规避 dynamic-loop 反向梯度的静默错误风险；
- 梯度双通路：**前向 JVP**（dual kernel，3–5 参数各 seed 一次，免轨迹存储，默认优化通路）与 **adstack 反向**（`qd.init(adstack=True)`，验证与多参数扩展）；两通路与中心差分在 T13 三路互验——**单轨下这是对 adstack 静默错误梯度的唯一防线**；
- ON-OFF 分支用 `@qd.ad.grad_replaced` 标记零梯度，仅前向仿真；
- 全部实数计算 float64；打印提示一律英文；
- **数学正确性兜底**：kernel 逐项对照 plan_a 文档 §3/§4 公式编写并 review，验证职责由解析锚点（T1/T4）与差分梯度（T5/T13）承担。

## 3. 接口契约（三层，v2 复查修订）

| 层 | 输入 | 输出 |
|---|---|---|
| L1 RHS `v̇ = f(x, θ, t)` | 状态 `(x₁,x₂,v₁,v₂)`、路面求值 `(z_r(t), ż_r(t))`、参数 θ、阻尼分支开关 | `(v̇₁, v̇₂)` + 导出 `F_d`、clip 激活掩码 |
| L2 轨迹（RK4 展开） | 初值 `x₀`（默认 0 = 静平衡）、路面对象（**detach**）、θ（唯一梯度路径） | 状态轨迹 `(B,T)`、`v̇₂`、`x₂−x₁`、`F_d`、`F_t = μω_t²(x₁−z_r)`、clip 掩码、`c(t)`（ON-OFF） |
| L3 指标 | L2 输出（无外部输入） | `J_acc = RMS(v̇₂)`、`J_travel = ‖x₂−x₁‖∞`、`J_tire = RMS(F_t)` → 可加权组合为标量 `J(θ)` |

**接口细则（2026-10-27 复查新增）**：
1. **路面求值接口**：`Road.eval(t) → (z_r, ż_r)`。D1 的正弦扫频/凸块用**解析闭式**（任意时刻可求值），保证 RK4 中间级 `t + dt/2` 安全取值；后续离散路谱以 dt/2 细网格 + 线性插值实现同一接口。**禁止**在 kernel 内对离散序列差分求 `ż_r`。
2. **初始瞬态**：`x₀ = 0`（静平衡）为默认；路面时程起始 **0.5 s ramp-in**（渐入包络），避免初始突变污染稳态指标。
3. **诊断输出**：clip 分支输出激活掩码（`F_sky` 越界标志，供 T9/T13 分域统计与优化诊断）；ON-OFF 分支输出 `c(t)` 时程作对照记录。
4. **J_acc 离散定义**：RMS 在**整步时刻记录的 `v̇₂(t_n)` 时程**上计算（不用 RK4 内部级值），全项目统一口径。
5. **batch 约定**：状态、路面、指标全 `(B, ·)` 批量化；路面 per-batch 可不同。
6. **参数约束参数化**：正性/序约束（`γ_sky ≥ 0`、`0 ≤ c_min < c_max`、`μ > 0`）在参数容器层以 `exp`/softplus 域实现，接口外不暴露无约束原值。
7. **层间输入语义**："输入"列仅列**层间外部输入**（路面、初值、参数等）；每层的隐含输入是上游层输出（L1←路面对象+参数容器，L2←L1，L3←L2）。L3 是纯归约层：`(B,T)` 时程 → `(B,)` 标量指标 → 可选组合标量 `J(θ) = w_acc·J_acc + w_travel·J_travel + w_tire·J_tire`（D5 冒烟先单指标 `J_acc`）。

三阻尼分支共用同一 RHS 骨架：`ideal = −γ_sky·v₂`；`clip = clip(γ_sky·v₂, F_lo, F_hi)`（F_lo/F_hi 由 `c_min/c_max·v_rel` 取 min/max）；`onoff = c(t)·v_rel`（符号判据）。参数与规范以 plan_a 文档 §2、§5 为准。

## 4. 阶段计划（D0 → D5，增量铁律：每阶段单变量推进、全绿才进下一阶段）

**D0 · 工具链验证**
安装锁定版本 quadrants；本机编译后端 hello-ODE 冒烟；`qd.init(adstack=True)` 已知梯度小样对照。
产出：环境文件（版本锁定）+ 冒烟脚本。Gate：编译后端 hello-ODE 跑通 + adstack 梯度对照通过。

**D1 · 被动基线（模型 I 退化：γ_sky = 0）**
状态/参数容器、RHS 骨架（Quadrants 单实现）、RK4 积分器、正弦/凸块路面。
Gate：T1、T2、T3。

**D2 · 模型 I 完整版**
仅加 `−γ_sky·v₂` 一项。
Gate：T4、T5、T6。

**D3 · 模型 II + ON-OFF 分支**
替换阻尼分支为力域 clip；加 ON-OFF（零梯度标记）。
Gate：T7、T8、T9、T13。

**D4 · 批量化与积分器对照**
batch 维、GPU 通路（CUDA 沙箱外）、半隐式 Euler 对照后端。
Gate：T10、T11、T13（batch/GPU 工况）。

**D5 · 优化闭环冒烟**
单变量 γ_sky 下降（J_acc，单凸块）→ 三变量 (γ_sky, c_min, c_max)。默认前向 JVP 通路。
Gate：T12。

## 5. 测试矩阵（13 项验收）

| # | 阶段 | 测试 | 通过标准 |
|---|---|---|---|
| T1 | D1 | 解析频响对拍：线性闭式 \|X₂/Z_r\|(ω) vs 长时正弦稳态 | 幅值误差 < 1e-3 |
| T2 | D1 | 模态频率：脉冲响应 FFT 峰 vs ω_s/2π、ω_t/2π | 偏差 < 2% |
| T3 | D1 | 能量审计：被动系统机械能单调衰减 | dE/dt ≤ 0（容差内） |
| T4 | D2 | ζ_eff 解析对照：自由衰减包络 vs (γ_s+γ_sky)/(2ω_s) | 偏差 < 5% |
| T5 | D2 | 梯度正确性：∂J/∂γ_sky 中心差分 vs 自动微分 | 相对误差 < 1e-5 |
| T6 | D2 | 天棚独占性：扰动 γ_sky 仅改 v̇₂ 行 | 结构断言 |
| T7 | D3 | 无源性审计：F_d·v_rel ≥ 0 全时程（含 clip 边界） | 无违例 |
| T8 | D3 | 理想极限：c_min/c_max 放开时模型 II ≡ 模型 I | RMS 差 < 1e-8 |
| T9 | D3 | 梯度分域：clip 界内梯度 = 模型 I 梯度；界上行为记录 | 一致 + 记录 |
| T10 | D4 | 积分阶：RK4 观测阶 ≈ 4；半隐式 Euler ≈ 1 | ±0.3 |
| T11 | D4 | batch 维展开一致、CPU-GPU 对拍 | 逐元素 < 1e-12 |
| T12 | D5 | 优化冒烟：J(θ) 下降、方向导数 FD 验证 | 单调段 + 方向正确 |
| T13 | D3–D4 | **梯度三路验证**（同后端内）：前向 JVP vs adstack 反向 vs 中心差分 | 相对误差 < 1e-5，clip 界内外分域 |

解析锚点：T1、T4（模型正确性）；可微性关卡：T5、T8、T9、T13（梯度正确性，兼防 adstack 静默错误）。单轨无独立参考实现：kernel 数学由"逐项对照 plan_a 公式 review + 解析锚点 + 差分梯度"三层兜底。

## 6. 交付物

- `skyhook/`：`road.py`、`model.py`（三分支 RHS，Quadrants 单实现）、`sim.py`（RK4 主、半隐式 Euler 对照）、`metrics.py`、`params.py`
- `tests/`：pytest，按阶段分组；每阶段验证记录（实测值 vs 通过标准）
- 环境文件：quadrants/torch 版本锁定

## 7. 风险与预案

| 风险 | 预案 |
|---|---|
| adstack 关闭时动态循环反向梯度**静默错误** | 宿主层循环规避 + T13 梯度三路验证专关；反向通路必须 `qd.init(adstack=True)` |
| clip 边界梯度截断致优化震荡 | 默认 straight-through；震荡则切 soft-clip（D3 预留开关） |
| Quadrants 版本迭代快（1.0→1.3.x） | 版本锁进环境文件；升级需全量回归（T1–T13） |
| 轮跳频段对 dt 敏感 | dt=1 ms 锁定；T10 过后不动 dt |
| 单轨无独立参考实现，kernel 数学错误滞留风险 | 代码逐项对照 plan_a §3/§4 公式 review；解析锚点 T1/T4 + 差分 T5/T13 兜底 |
| 单点依赖 Quadrants 工具链（编译/后端阻塞） | 阻塞时向用户回报并重议框架决策，不擅自加轨 |

## 8. 工程纪律（全程生效）

1. 增量式开发：每阶段只加一个变量，通过再扩面；
2. 打印提示一律英文；
3. CUDA 程序沙箱外运行（`dangerouslyDisableSandbox` 启动，防静默回退 CPU）；
4. 路面输入全程 detach；F_d 保持显式中间节点（第三律配对）；
5. float64；RK4 dt = 1 ms；正弦稳态对拍用解析稳态解免长瞬态窗。

## 9. 里程碑

- M0 = D0 通过（工具链就绪）
- M1 = D1 通过（被动基线 + 三轨对拍）
- M2 = D2 通过（模型 I 全绿）
- M3 = D3 通过（模型 II + ON-OFF 全绿）
- M4 = D4 通过（批量化/GPU）
- M5 = D5 通过（优化闭环）→ 进入 co-design 阶段

---

## 10. 执行结果闭环（D0–D5 验收记录，2026-10-27）

> 本节落档 §6 交付物所要求的"每阶段验证记录（实测值 vs 通过标准）"。D0–D5 全部执行完毕，测试矩阵 T1–T13 全部通过，全量回归 **47 passed（9.78 s，x64 + CUDA，无 skip / xfail）**，里程碑 **M0–M5 全部达成**，计划 A 收官。环境：quadrants 1.3.0（llvm 22.1.0，commit ab9a58ab），python 3.12.13，conda env `diffphy`（environment.yml 锁定）。

### 10.1 各阶段 Gate 验收汇总

| 阶段 | Gate | 通过标准 | 实测 | 判定 |
|---|---|---|---|---|
| D0 | 工具链 | 编译后端 hello-ODE 跑通 + adstack 梯度对照 | hello-ODE RK4 vs 解析解：x_err 8.521e-13 / v_err 1.025e-09；标量梯度 3/3 精确（1.0 / 6.0 / −2.5）；动态循环求和梯度 max_err 0.0 | ✓ |
| D1 | T1/T2/T3 | 见 §5 | T1 幅值误差 2.6e-06；T2 频率偏差 0.13% / 1.16%；T3 能量单调衰减 + 耗散平衡 rel err 1.4e-10 | ✓ |
| D2 | T4/T5/T6 | 见 §5 | T4 ζ_eff 偏差 0.19%；T5 AD vs FD rel 1.2e-08；T6 v̇₁ 行位级不变、v̇₂ 差 = −γ_sky·v₂ 精确 | ✓ |
| D3 | T7/T8/T9/T13 | 见 §5 | T7 min F_d·v_rel = 0（无违例）；T8 rel 3.1e-16；T9 四域样本三路全对；T13 三参数 rel ≤ 6.4e-09 | ✓ |
| D4 | T10/T11/T13 | 见 §5 | T10 观测阶 4.006 / 1.010；T11 批量展开 diff = 0.0、CPU–GPU 最大 9.7e-14；T13 批量三路 rel ≤ 2.4e-09（含 GPU） | ✓ |
| D5 | T12 | 单调段 + 方向正确 | 单变量 20/20 迭代严格降（−3.7%）；三变量 25/25（−2.1%）；方向导数 rel 6.1e-11 / 3.6e-09 | ✓ |

各阶段测试数：D0 5、D1 9、D2 7、D3 15、D4 9（含 GPU 2）、D5 2，合计 **47**。复验命令（项目根，GPU 用例按纪律沙箱外运行）：`python -m pytest tests/ -q`。

### 10.2 测试矩阵实测明细（T1–T13）

| # | 通过标准 | 实测 | 判定 |
|---|---|---|---|
| T1 | 幅值误差 < 1e-3 | 29 频点最差相对误差：\|X₁/Z\| 1.799e-06、\|X₂/Z\| 2.616e-06 | ✓ |
| T2 | 偏差 < 2% | 车身模态 FFT 峰 1.3919 Hz vs 解析 1.3937 Hz（0.13%）；车轮模态 12.7661 vs 12.9154 Hz（1.16%） | ✓ |
| T3 | dE/dt ≤ 0（容差内） | E 2.054665e-03 → 3.160720e-13，max step dE = −1.731e-19；耗散积分平衡 rel err 1.430e-10 | ✓ |
| T4 | 偏差 < 5% | 纯天棚 ζ_eff 实测 0.299443 vs 公式 0.300000（rel 1.857e-03）；阻尼工况衰减率 vs QEP 根 rel 6.516e-10 | ✓ |
| T5 | 相对误差 < 1e-5 | dJ/dγ_sky：AD −2.9656532917e-02 vs FD −2.9656533270e-02，rel 1.192e-08 | ✓ |
| T6 | 结构断言 | 扰动 γ_sky 后 v̇₁ 行位级不变（bit-identical，3 组样本），v̇₂ 差 = −γ_sky·v₂ 逐位精确 | ✓ |
| T7 | 无违例 | clip / ON-OFF 两律 min F_d·v_rel = 0.000e+00（1000 步，含 clip 边界）；自由衰减耗散积分 2.2498e-04 ≥ 0，E 2.860211e-04 → 3.413e-12，max step dE = −1.547e-17 | ✓ |
| T8 | RMS 差 < 1e-8 | 被动退化 rel = 3.070e-16；理想可达性：非激活步 608/1000 上 F_d = F_sky 位级精确，激活步 392（结构性极限，如实记录） | ✓ |
| T9 | 一致 + 记录 | 4 个分域样本（界内 / 下裁 / 上裁 / 负 v_rel）：反向 / JVP / FD 三路全部命中解析期望；混合域激活 666/1000 | ✓ |
| T10 | 观测阶 ±0.3 | RK4：误差 3.299e-08 → 4.952e-13（dt 五级减半），平均阶 **4.006**；半隐式 Euler：8.829e-05 → 5.362e-06，平均阶 **1.010** | ✓ |
| T11 | 逐元素 < 1e-12 | 批量 vs 逐条展开：全通道 diff = 0.000e+00，j vs numpy RMS 2.220e-16；CPU–GPU 对拍最大逐元素差 9.681e-14（ft 通道），mask 位级 0.0，grad_ad 3.469e-18 | ✓ |
| T12 | 单调段 + 方向正确 | 单变量 γ_sky：20/20 迭代严格降，J 1.73359664 → 1.66971008（rel −3.685e-02）；三变量：25/25，J 1.67012722 → 1.63444356（rel −2.137e-02）；方向导数 FD vs JVP rel = 6.072e-11 / 3.627e-09，方向均为负 | ✓ |
| T13 | rel < 1e-5，clip 界内外分域 | 单轨（D3）：γ_sky 5.038e-11、c_min 6.411e-09、c_max 3.025e-10（AD/FD），AD/JVP ~1e-15；批量 CPU（D4）：b∈{0,3,6} rel 1.1e-10 ~ 4.8e-10，梯度向量 8 元素全对（max 2.42e-09）；GPU 三路 rel ≤ 1.09e-09；激活集稳定性断言全 True | ✓ |

T12 补充：单变量 γ_sky 1.4137 → 2.5408；三变量 θ = (γ_sky, c_min, c_max) 自 (2.8274, 1.1310, 2.8274) → (3.5193, 0.7605, 3.2507)（γ_sky↑、c_min↓、c_max↑，物理方向合理）；两工况均以 n_iters 自然终止（stopped = "max_iter"），rollout 评估次数 61 / 126（全程 < 1 s），全部迭代点满足投影约束。ON-OFF 零梯度标记实测：切换律 1000/1000 步与符号判据一致（高阻尼 732 步），Tape 下参数梯度精确 (0, 0, 0)。

### 10.3 里程碑达成

- **M0–M5 全部达成**：M0 工具链就绪 → M1 被动基线 → M2 模型 I 全绿 → M3 模型 II + ON-OFF 全绿 → M4 批量化/GPU → M5 优化闭环。
- 计划 A 就此收官；后续 co-design（NN 策略 / 多目标加权 J(θ)）另行立项，不在本文档范围。

### 10.4 实际交付物清单（vs §6）

| §6 计划交付物 | 实际落位 | 备注 |
|---|---|---|
| `skyhook/road.py` | ✓ | ZeroRoad / SineRoad / BumpRoad，解析闭式求值（细则 1），road_eval 为 model/model2 共用单一真源 |
| `skyhook/model.py`（三分支 RHS） | ✓（拆分，见 §10.6-2） | 模型 I kernels：rk4_step、euler_step（D4 对照）、rollout_record、probe_rhs |
| `skyhook/sim.py`（RK4 主、Euler 对照） | ✓ | run_passive / run_passive_euler |
| `skyhook/metrics.py` | —（并入，见 §10.6-3） | J_acc 以 kernel 归约（model2/model2b）+ 宿主 numpy RMS 双实现互验（T11：2.2e-16） |
| `skyhook/params.py` | ✓ | PassiveParams / SemiActiveParams（默认值同 plan_a §5） |
| `tests/` 按阶段分组 | ✓ | test_d0…test_d5 + analytic.py / d4_fixture.py / gpu_worker.py（共享解析锚点与批量夹具） |
| 每阶段验证记录 | ✓ | 即本节 §10 |
| 环境文件 | ✓ | environment.yml（quadrants 1.3.0 / python 3.12.13） |
| （计划外）`model2.py` / `sim2.py` | ✓ | D3：模型 II clip + ON-OFF kernels 与宿主层 |
| （计划外）`model2b.py` / `sim2b.py` | ✓ | D4：批量化（B_MAX=64，T_B_MAX=4096，前导 B 字段） |
| （计划外）`opt.py` | ✓ | D5：投影 Armijo PGD + FwdMode JVP 梯度通路 |
| （计划外）`_runtime.py` | ✓ | ensure_init 幂等后端绑定（arch 单次锁定） |

### 10.5 执行期关键工程发现（已固化进代码注释与项目记忆）

1. **反向 AD 三条铁律**（D2 确立，D4 批量下复验成立）：整时间循环留在单 kernel 内 + 状态历史槽位每槽只写一次 + `for _ in range(1)` 外包裹使动态范围 adstack 合法；
2. **激活集稳定性锚点**（D3）：严格无 clip 的非平凡轨迹不存在（相位解锁，v₂/v_rel 扫满实轴），T9/T13 三路一致性以 ±h 扰动下 mask 位级不变为光滑性前提；
3. **FD 步长陷阱**（D3）：整步 mask 稳定 ≠ RK4 stage 级稳定，全回路三路验证统一 h = 1e-4；
4. **body_mode_ic 特征向量修正**（D4）：行 2 本征关系分母应为 γ_s·s + ω_s²（原 ω_s² 仅 γ_s = 0 时正确）；修正后 T10 的 RK4 收敛阶从 ~7e-7 误差地板恢复至 4.006；
5. **sqrt(0) 对偶污染**（D4）：J = RMS 的对偶在 J = 0 批元处为 0/(2√0) = nan 并毒化整条 dual 链——平路（ZeroRoad）批元必须给非零初条件；
6. **FwdMode N-D seed 全长规则**（D4）：seed 长度必须等于参数字段全长（B_MAX = 64），而非有效批量 B；
7. **GPU worker 模式**（D4）：子进程先 `ensure_init(arch=qd.cuda)` 再 import 模型模块，stdout 断言 `arch=cuda` 防静默回退 CPU；pytest 与 worker 均须沙箱外运行；
8. **ON-OFF 零梯度标记**（D3）：`qd.ad.grad_replaced` 包装下 Tape 参数梯度精确 0.0，杜绝穿透 qd.select 不连续分支的静默错误梯度。

### 10.6 偏差记录（与原文差异及理由）

1. **约束处理**：§3 细则 6 原定 exp/softplus 域参数化，D5 实际采用**投影 Armijo PGD**（`_project`：γ_sky ≥ 0、c_min ≥ 0、c_max ≥ c_min + c_gap，c_gap = 0.05）——保持物理参数直接可读，投影语义与 clip 截断次梯度 AD 天然兼容；co-design 阶段如需无约束优化器可再切回参数化路径；
2. **模块拆分**：§6 原列单文件 model.py 承载三分支 RHS，实际按阶段演进拆为 model.py（模型 I）/ model2.py（模型 II + ON-OFF）/ model2b.py（批量）及对应宿主层，§3 接口契约未变；
3. **metrics.py 并入**：J_acc 归约以 kernel + numpy 双实现互验（T11），优于单独模块；
4. **测试粒度**：13 项矩阵细化为 47 个 pytest 用例（矩阵行 → 参数化 / 分域 / 通道展开），通过标准未放松；
5. **M1 措辞**："三轨对拍"为 v1.0 遗留表述，实际按决策 ④ 执行（解析锚点 + 差分梯度兜底，无独立参考实现）。
