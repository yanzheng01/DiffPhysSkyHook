# 参数说明文档：动力学模型参数 × Plan C 训练设定参数

> 版本：v1.6（2026-11-02，新增 §4.5 域随机化设计：随机化轴/冻结轴/设计原则/时序默认值；同步 plan_c v2.3）
> 依据：[params.py](../skyhook/params.py)、[model.py](../skyhook/model.py)、[model2b.py](../skyhook/model2b.py)、[road.py](../skyhook/road.py)、[opt.py](../skyhook/opt.py)（代码实测默认值）+ [plan_c_policy_training_dev_test_plan.md](./plan_c_policy_training_dev_test_plan.md) v2.3（设计值）
> 标记约定：✅ = 已实现并有测试覆盖；📋 = Plan C 设计值（未实现，以 plan_c 文档为准）

---

## 1. 量纲规范（m₂ ≡ 1）

四分之一车采用质量归一化：簧载质量 m₂ 归一为 1，全部阻尼/刚度以"增益"表达（力 = 增益 × 速度/位移）。**位移 x₁, x₂, z_r 不经归一，保持物理米数**；所有增益的单位是 m₂=1 规范下的等效 SI。

| 派生锚点 | 公式 | 默认值 |
|---|---|---|
| 车身模态频率 | ω_s/2π, ω_s = √ω_s² | 1.5 Hz |
| 轮跳模态频率 | ω_t/2π, ω_t = √ω_t² | 12.0 Hz |
| 车身阻尼比 | ζ = γ/(2ω_s) | 见各参数行 |

---

## 2. 动力学模型参数（✅ 已实现）

### 2.1 被动参数 PassiveParams（模型 I，[params.py:18-71](../skyhook/params.py#L18-L71)）

| 参数 | 默认值 | 物理含义 | 锚点 |
|---|---|---|---|
| `mu` | 0.1 | 质量比 m₁/m₂ | — |
| `omega_t_sq` | (2π·12)² ≈ 5684.9 | 轮胎刚度增益 k₁/μ | 轮跳 12 Hz |
| `omega_s_sq` | (2π·1.5)² ≈ 88.83 | 悬架弹簧增益 k₂ | 车身 1.5 Hz |
| `gamma_t` | 2·0.02·2π·12 ≈ 3.016 | 轮胎阻尼增益 c₁/μ | ζ_t ≈ 0.02 |
| `gamma_s` | 2·0.10·2π·1.5 ≈ 1.885 | 悬架阻尼增益 c₂（被动） | ζ_s ≈ 0.10 |
| `gamma_sky` | 0.0 | 理想天棚增益（模型 I 专用） | 默认 = 被动基线（D1） |

约束（`__post_init__` 强制）：mu > 0；刚度增益 > 0；阻尼增益 ≥ 0；全部有限。`from_unconstrained` 提供 exp 域映射（优化器接口契约细则 6）。

### 2.2 半主动参数 SemiActiveParams（模型 II，[params.py:74-127](../skyhook/params.py#L74-L127)）

被动四参数同 §2.1，另有三参数替代 gamma_s：

| 参数 | 默认值 | 物理含义 | 锚点 |
|---|---|---|---|
| `gamma_sky` | 2·0.30·2π·1.5 ≈ 5.655 | 理想天棚力指令增益 | ζ_sky ≈ 0.30 |
| `c_min` | 2·0.03·2π·1.5 ≈ 0.565 | 力窗下限（软） | ζ(c_min) ≈ 0.03 |
| `c_max` | 2·0.30·2π·1.5 ≈ 5.655 | 力窗上限（硬） | ζ(c_max) ≈ 0.30 |

可实现性力窗：`F_d = clip(γ_sky·v₂, min(c_min,c_max)·v_rel, max(c_min,c_max)·v_rel)`（[model2.py](../skyhook/model2.py) `fd_clip_cmd`）。约束：0 ≤ c_min ≤ c_max；`from_unconstrained` 用 softplus 映射（γ_sky、c_min ≥ 0；c_max = c_min + softplus(shift)），硬件增益 mu/ω_t²/ω_s²/γ_t 固定不参与优化。

### 2.3 数值参数（✅）

| 参数 | 默认值 | 出处 | 说明 |
|---|---|---|---|
| `dt` | 1.0e-3 s (1 ms) | [sim.py](../skyhook/sim.py)/[sim2.py](../skyhook/sim2.py)/[sim2b.py](../skyhook/sim2b.py) 签名 | 积分步长，锁定不调 |
| `t_end` | 2.0 s | 同上 | 默认仿真时长（n_steps = t_end/dt = 2000） |
| `n_steps` | 1000（优化器默认） | [opt.py:118](../skyhook/opt.py#L118)/[opt.py:228](../skyhook/opt.py#L228) | 优化窗口 1 s |
| `T_MAX` | 262144 | [model.py:31](../skyhook/model.py#L31) | 标量通道记录槽上限（256 s） |
| `B_MAX` | 64 | [model2b.py:38](../skyhook/model2b.py#L38) | 批量通道批元上限 |
| `T_B_MAX` | 4096 | [model2b.py:39](../skyhook/model2b.py#L39) | 每批元记录槽上限（n_steps ≤ T_B_MAX−1） |
| 积分器 | RK4 | 各 rollout kernel | 半隐式 Euler 仅作收敛阶对照 |
| 浮点 | float64 | 所有 qd.field | 精度纪律 |
| 初值 `x0` | (0,0,0,0) | sim2.py 签名 | 默认静止；**Plan C 训练须非零 IC**（sqrt(0) 对偶告诫，见 §4.4） |

### 2.4 路面参数（✅，[road.py](../skyhook/road.py)）

解析闭式（任意 t 可求值，含 RK4 半步；对设计参数梯度 detach）：

| 路面 | 参数 | 说明 |
|---|---|---|
| ZeroRoad | — | 平路（自由振动/衰减测试） |
| SineRoad | `amp`（幅值 m）、`freq_hz`（频率 Hz）、`ramp_t`（默认 0.5 s） | z = amp·sin(2πf·t)·e(t)，smoothstep ramp-in（C1 包络，防 t=0 冲击） |
| BumpRoad | `amp`（m）、`t0`（起点 s）、`width`（宽度 s） | 半正弦凸块，[t0, t0+width] 外为 0 |

### 2.5 Plan C 新增仿真参数（📋 设计值，plan_c v1.2 决策 8）

| 参数 | 设计值 | 说明 |
|---|---|---|
| `x_max_b` | (B,) 批量字段，默认全批元 0.08 m | 行程软边界；复用 per-batch 增益模式，无 needs_grad；位移不经质量归一，直接物理米数 |
| 离散路面 | 宿主侧 (z, ż) 半步网格 + kernel 线性插值 | 替代/并列解析路面（road_eval_seq）；kernel 内禁止差分求 ż |

---

## 3. Plan A/B 优化器参数（✅ 已实现，[opt.py](../skyhook/opt.py)）

| 参数 | Plan A（`optimize`，硬件参数） | Plan B（`optimize_force`，力序列） |
|---|---|---|
| 方法 | 投影梯度下降 + Armijo 回溯 | 无投影梯度下降 + Armijo 回溯 |
| `n_iters` | 20 | 20 |
| `alpha0`（初始步长） | 1.0 | 10.0 |
| `c1`（Armijo 常数） | 1e-4 | 1e-4 |
| `shrink`（回溯收缩因子） | 0.5 | 0.5 |
| `alpha_min` | 1e-6 | 1e-8 |
| 投影域 | softplus 参数化（γ_sky, c_min, c_max） | 无投影（模型侧 clip 保证可实现性） |
| 梯度通路 | 前向 JVP（默认） | 反向 Tape（默认），一次全 (T,)/(B,T) 伴随 |

---

## 4. Plan C 训练设定参数（📋 设计值，plan_c v1.2）

### 4.1 损失权重

| 权重 | 设计值 | 损失项 | 定义 |
|---|---|---|---|
| `w_acc` | 1.0 | L1 车身加速度 | MS(v̇₂)，主目标 |
| `w_disp` | 0.1 | L2 悬架行程 | MS(x₂−x₁) |
| `w_tire` | 0.1 | L3 轮胎动载荷 | MS(F_t) |
| `w_lim` | 0.5 | L6 行程软边界 | MS(ReLU(‖x₂−x₁‖−x_max))，界内精确为 0 |
| `w_jerk` | 0.01 | L4 控制平滑 | MS(Δf_cmd/Δt)，torch 侧 |
| `w_mag` | 0.001 | L5 控制幅值 | MS(f_cmd)，torch 侧 |

总损失 J = J_sim(L1+L2+L3+L6, kernel 侧 Tape 伴随) + J_ctl(L4+L5, torch autograd)。权重组为 F2 冒烟调定前的初始猜测；RMS/max 行程/越界率仅作评测指标。

### 4.2 优化器与训练回路

| 参数 | 设计值 | 出处/说明 |
|---|---|---|
| 优化器 | AdamW + CosineAnnealingLR | 借鉴 DiffPhysDrone |
| 伴随时间衰减 λ | 默认 1.0（消融 0.97） | 段内细调旋钮（段间 detach 已提供信用截断）；桥 backward 内乘 λ^n |
| 输出层初始化 | 权重 ×0.01 + 非零偏置 | 小初始化；偏置取 passive 等效 c_min·v_rel 量级（避开零指令平台点） |
| 梯度合成 | `loss.backward()` 一次 | sim 部分经 torch↔Quadrants 桥（fc_b.grad），ctl 部分走 torch 原生链 |

### 4.3 窗口与批量

| 参数 | 设计值 | 说明 |
|---|---|---|
| episode / 分段 | T_w = 1 s；K = 10 段 × H = 0.1 s（100 步/段，均可配；冒烟 K=5×H=100）；**决策频率消融 H ∈ {100, 50, 25}**（10/20/40 Hz） | 分段滚动规划（plan_c v2.0 决策 1）：段边界 10 Hz 推理，段间状态 detach，单 kernel 段 rollout + 一次 Tape；频率层级：仿真/指令波形 1 kHz、决策 10 Hz——闭环修正带宽以车身模态（1.5 Hz）为限，轮跳（12 Hz）靠前馈与鲁棒性（v2.2） |
| 批量 B | 冒烟 8，训练 ≤ 64（B_MAX） | |
| 模式 1 观测 | 过去激励窗 a₂（0.5 s → 50 点，加速度计无预览，σ_a 默认 0）+ 过去指令窗 fc（50 点）+ 当前状态 o_c 5 维 (x₁,x₂,v₁,v₂,x_max) | plan_c 决策 9；固定 scale 归一化；10 ms mean-pooling |
| 模式 2 观测 | 模式 1 全部通道 + 前视预览窗 **(z_evt, ż_evt) 事件分量**（未来 0.2 s → 40 点 × 2ch，**5 ms 池化**，仅含坑洞/凹陷/凸起等几何事件、不含材料粗糙度分量，v1 oracle 真值） | 公平协议：模式 1 = 预览置零的模式 2（同参数量）；前向传感器仅见事件，粗糙度两模式同盲（负对照） |
| 域随机化 | 详见 §4.5（7 个随机化轴 + 冻结轴清单） | 总纲："环境进分布、设备不进、梯度不穿" |

### 4.4 纪律性约束（继承 + 新增）

| 约束 | 说明 |
|---|---|
| 非零 IC | 所有训练/评测批元 IC 非零或路面非零（sqrt(0) 对偶污染，D4 发现） |
| 非零指令初值 | 零指令是严格平台点（伴随精确全零），初始输出偏置非零 |
| setup 在 AD 上下文外 | 参数/指令/IC 写入一律在任何 Tape/FwdMode 之外 |
| 原有 kernel 不改只增 | model/model2/model2b 冻结，增量全部在 model3 |
| 回归门 | Plan A/B 67 用例每阶段全绿 |

### 4.5 域随机化设计（plan_c 决策 7/10，📋 设计值）

**总纲**：环境进分布、设备不进、梯度不穿——随机化只施加在环境/场景轴，设备与硬件参数冻结（co-design 语义），全部随机量经宿主侧种子生成后 detach 写入。

**随机化轴（per-batch）**：

| # | 轴 | 随机内容与默认范围 | 适用模式 | 梯度 |
|---|---|---|---|---|
| 1 | 路面族 | 批内混合采样（AR 平整族 / AR+事件族按模式） | 两模式 | detach |
| 2 | 粗糙度等级 | σ 分档随机（沥青级↔混凝土级谱） | 两模式 | detach |
| 3 | 粗糙度切换 | t_s ∈ (0.2, 0.8) s 均匀（建议值，F0 落档），1 次/episode | 两模式 | detach |
| 4 | 事件参数 | 符号（凸/坑/凹）、幅值、宽度 ≥ 0.05 s、间距 ≥ 0.15 s、1–3 个/episode（建议值） | 模式 2 | detach |
| 5 | 初始状态 IC | per-batch 随机，保持非零纪律 | 两模式 | detach |
| 6 | 传感器噪声 σ_a | 加速度计加性噪声，默认 0，F2 扫描 {0, 0.05, 0.1} | 两模式 | detach |
| 7 | 行程包络 x_max_b | **预留**批量随机化（默认固定 0.08；启用时同步进观测第 7 维） | 预留 | detach |

**冻结轴**：力窗 (c_min, c_max)（决策 7，沿 Plan B 决策 4）、硬件增益 (μ, ω_t², ω_s², γ_t)、架构频率（dt/K/H，H 为消融旋钮非随机轴）、损失权重 w 组。

**设计原则**：① 种子确定性（训练可复现；F3 泛化评测用 held-out 种子族切分未见粗糙度/事件组合）；② 设备/环境二分（"车上不变的"不随机、"世界会变的"随机——DiffPhysDrone 同源哲学）；③ 随机化 ≠ 梯度（随机轴只影响前向分布，∂J/∂f_cmd 链不穿）；④ 对照保护（T28 两模式同种子同分布，差异只在预览通道；粗糙度切换两模式同盲，负对照成立）。

**不在本轮**：前向传感器退化（预览噪声/截断视程，F3 后消融）、作动器动力学（范围外，若加入为新增轴）。

---

## 5. 参数速查：谁进梯度、谁冻结

| 类别 | 参数 | 进梯度？ |
|---|---|---|
| 可优化（Plan A） | γ_sky, c_min, c_max | ✅ JVP/Tape，softplus 域 |
| 可优化（Plan B） | f_cmd[n] | ✅ Tape，无投影 |
| 可优化（Plan C） | 策略网络权重 θ | ✅ AdamW，经桥伴随 + torch 原生 |
| 设备参数（冻结） | mu, ω_t², ω_s², γ_t, (c_min, c_max) 力窗 | ❌（Plan A 优化时 γ_sky/c_min/c_max 除外） |
| 硬件包络（冻结） | x_max_b（Plan C 新增） | ❌ 无 needs_grad，预留批量随机化 |
| 环境输入（detach） | 路面 z_r/ż_r（解析或离散）、IC | ❌ 天然 detach |
