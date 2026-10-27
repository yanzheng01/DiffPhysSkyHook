# 方案 A：质量归一化天棚动力学模型

> 版本：v1.0（2026-10-27）
> 上游依据：[skyhook_model_research.md](./skyhook_model_research.md) §2.1、§4；二自由度悬架示意图（m₁/m₂ 双质量，k₁-c₁ 接地、k₂-c₂ 跨接，谐波力或路面激励）
> 适用范围：可微仿真、参数梯度优化、天棚族控制律对比

---

## 1. 设计动机与界定

**方案 A 的定义**：将牛顿形式 $m\ddot{x} = \sum F$ 重写为加速度增益形式 $\ddot{x} = \sum (F/m)$，消去质量与力的独立单位，只保留各元件的"单位质量增益"（$k/m$、$c/m$）。

**无损性**：决定系统响应的本征量是比值——$\omega_n^2 = k/m$、$\zeta = c/(2\sqrt{km})$。将全部质量与力同乘 $\alpha$，加速度动力学不变，因此绝对质量标度在运动方程中不可辨识；固定规范（本文取 $m_2 \equiv 1$）后，模型与原始力-质量形式**严格等价**，仅是重参数化。

**与不可行方案的边界**：本方案不允许把加速度当作给定输入（跳过力求解的"运动学回放"）。天棚力依赖速度、速度是加速度的积分、加速度由力求解——该反馈闭环必须保留，否则阻尼、共振与参数梯度全部消失。

## 2. 坐标、规范与耦合约束

- 状态：绝对位移 $x_1$（非簧载，对应图 m₁）、$x_2$（簧载，对应图 m₂）；绝对速度 $v_1, v_2$。
- 输入：路面高程 $z_r(t)$（预给定时程，进入计算图但 detach，不参与梯度）。
- **规范（gauge）**：$m_2 \equiv 1$（固定，不学习），$m_1 = \mu$。质量比 $\mu$ 是唯一的跨质量耦合标度。
- **第三律配对约束（勿破坏）**：悬架阻尼力 $F_d$ 必须保留为显式中间变量，以 $F_d/\mu$ 与 $-F_d$ 分别进入两个方程；悬架弹簧增益受 $\omega_s^2/\mu$ 约束。若将 $c/m_1$、$c/m_2$ 设为独立自由参数，即破坏等大反向条件，模型类超出物理悬架（可能出现"凭空注入动量"的假解）。

参数映射（图 → 悬架 → A 形式）：

| 图中符号 | 悬架物理量 | A 形式增益 | 量级锚点 |
|---|---|---|---|
| $m_2$（上质量） | 簧载质量 $m_s$ | $\equiv 1$（规范） | — |
| $m_1$（下质量） | 非簧载质量 $m_u$ | $\mu$ | 0.05–0.15 |
| $k_1$ | 轮胎刚度 $k_t$ | $\omega_t^2 = k_1/\mu$ | $(2\pi \cdot 12)^2 \approx 5700\ \mathrm{s^{-2}}$ |
| $k_2$ | 悬架弹簧 $k_s$ | $\omega_s^2 = k_2$ | $(2\pi \cdot 1.5)^2 \approx 89\ \mathrm{s^{-2}}$ |
| $c_1$ | 轮胎阻尼 | $\gamma_t = c_1/\mu$ | $\approx 0$ |
| $c_2$ | 悬架阻尼器 | $\gamma_s = c_2$；半主动时由 $c_{\min}, c_{\max}$ 界定 | $c_{\min} \approx 0.1\,c_{\max}$ |
| （新增） | 天棚阻尼 $c_{sky}$ | $\gamma_{sky} = c_{sky}/m_2$ | $0.5\text{–}2\,\zeta_s\omega_s$ 量级 |

## 3. 模型 I：理想天棚（分析基准）

$$
\begin{cases}
\dot{x}_1 = v_1, \qquad \dot{x}_2 = v_2 \\
\dot{v}_1 = -\gamma_t\,(v_1 - \dot{z}_r) - \omega_t^2\,(x_1 - z_r) - \dfrac{\omega_s^2}{\mu}(x_1 - x_2) - \dfrac{\gamma_s}{\mu}(v_1 - v_2) \\
\dot{v}_2 = -\,\omega_s^2\,(x_2 - x_1) - \gamma_s\,(v_2 - v_1) - \gamma_{sky}\,v_2
\end{cases}
$$

说明：$\gamma_t (v_1-\dot z_r)$ 即 $\frac{c_1}{\mu}(v_1-\dot{z}_r)$。天棚项 $-\gamma_{sky}v_2$ **只出现在 $\dot{v}_2$ 行**——虚拟阻尼器不向任何真实物体传递反作用力，在此形式下表现为独占项；这正是理想天棚需要主动作动器（进而被半主动策略逼近）的数学表达。

## 4. 模型 II：半主动连续律（可微仿真主用）

### 4.1 力域投影（推荐实现）

将固定 $\gamma_s$ 替换为投影阻尼力 $F_d$，采用**力域 clip**（避免 $v_{rel}\to 0$ 的除法奇异）：

$$
F_{sky} = \gamma_{sky}\,v_2, \qquad v_{rel} = v_2 - v_1
$$

$$
\underline{F} = \min(c_{\min} v_{rel},\, c_{\max} v_{rel}), \qquad
\overline{F} = \max(c_{\min} v_{rel},\, c_{\max} v_{rel})
$$

$$
F_d = \mathrm{clip}(F_{sky},\ \underline{F},\ \overline{F})
$$

动力学方程：

$$
\begin{cases}
\dot{x}_1 = v_1, \qquad \dot{x}_2 = v_2 \\
\dot{v}_1 = -\gamma_t\,(v_1 - \dot{z}_r) - \omega_t^2\,(x_1 - z_r) - \dfrac{\omega_s^2}{\mu}(x_1 - x_2) + \dfrac{F_d}{\mu} \\
\dot{v}_2 = -\,\omega_s^2\,(x_2 - x_1) - F_d
\end{cases}
$$

### 4.2 性质

- **无源性内嵌**：$F_d \cdot v_{rel} \ge 0$ 恒成立（clip 的上下界保证），模型不会供能；
- **次可微**：clip 在界内对 $\gamma_{sky}, c_{\min}, c_{\max}$ 梯度完整，在界上梯度截断（可用 straight-through 或 soft-clip 光滑化）；
- **理想极限**：当 $F_{sky}$ 始终落在 $[\underline F, \overline F]$ 内时，$F_d = F_{sky}$，模型 II 退化为模型 I。

### 4.3 ON-OFF 基线（仅用于对比，不参与梯度优化）

$$
c(t) =
\begin{cases}
c_{\max}, & v_2\,(v_2 - v_1) > 0 \\
c_{\min}, & \text{否则}
\end{cases}
\qquad F_d = c(t)\,v_{rel}
$$

切换不连续、梯度几乎处处为零，只作为性能下界参照（见研究报告 §4.6）。

## 5. 可学习参数与优化设置

| 参数 | 是否可学习 | 约束建议 |
|---|---|---|
| $\mu$ | 可选 | $\exp(\theta)$ 保证正性；固定 0.1 亦可 |
| $\omega_t^2,\ \omega_s^2$ | 可选（硬件给定则固定） | 正性约束；两端频率比 ≈ 8–10 |
| $\gamma_{sky}$ | 是（核心设计变量） | 非负 |
| $c_{\min},\ c_{\max}$ | 是 | $0 \le c_{\min} < c_{\max}$，用 $\log$ 域参数化 |
| $z_r(t)$ | 否 | 预给定时程，计算图中 detach |

评价指标三件套（与研究报告 §6.3 对齐）：

$$
J_{acc} = \mathrm{RMS}(\dot{v}_2), \qquad
J_{travel} = \max_t |x_2 - x_1|, \qquad
J_{tire} = \mathrm{RMS}\big(\mu\,\omega_t^2\,(x_1 - z_r)\big)
$$

其中轮胎动载荷按规范反推：$F_t = \mu \cdot \omega_t^2 \cdot (x_1 - z_r)$；阻尼器出力 $F_d$ 已是显式量；悬架动行程为纯状态差。

## 6. 数值实现要点

1. **积分器**：RK4 或半隐式 Euler；含 10–15 Hz 轮跳分量时 $dt \le 5\ \mathrm{ms}$（建议 1 ms）；
2. **正性参数化**：所有刚度/阻尼增益用 $\exp(\theta)$ 或 softplus 生成；
3. **奇异防护**：力域 clip 已消除 $v_{rel} = 0$ 奇异；如需 $c(t)$ 显式值，用 $c(t) = F_d / (v_{rel} + \epsilon\,\mathrm{sgn})$ 仅作记录、不进梯度；
4. **路面输入**：$z_r(t)$ 与 $\dot{z}_r(t)$ 由路谱生成（滤波白噪声 / 正弦扫频 / 凸块），对设计参数梯度 detach；
5. **梯度链路**：$J \leftarrow \dot{v}_2 \leftarrow F_d \leftarrow (\gamma_{sky}, c_{\min}, c_{\max})$，无除以状态量的环节，反向传播数值条件良好。

## 7. 等价性验证清单

- [ ] 置 $\gamma_{sky} = 0$、$c_{\min} = c_{\max} = \gamma_s$，模型 II 应退化为被动四分之一车，脉冲响应频率 ≈ $\omega_s/2\pi$ 与 $\omega_t/2\pi$；
- [ ] 理想模型 I 的车身模态阻尼比应满足 $\zeta_{eff} = (\gamma_s + \gamma_{sky})/(2\omega_s)$；
- [ ] 半主动模型 II 中校验 $F_d \cdot v_{rel} \ge 0$ 恒成立（无源性数值审计）；
- [ ] 与原始力-质量形式（研究报告 §2.1）在同一 $z_r$ 时程下逐步对拍，位移轨迹应一致到积分误差量级；
- [ ] 能量审计：$W = \int F_d \cdot v_{rel}\,dt \ge 0$（天棚只耗能）。

## 8. 与研究报告的衔接

- 原始力-质量模型与符号约定：报告 §2.1；
- 天棚族控制律（ON-OFF / 连续 / SH-ADD / Skygroundhook / VISKY）：报告 §4；
- 频段画像（收益区 / 发颠区 / 让权区）与本文增益量级的对应：报告 §4.6；
- 可微性讨论与"硬件即策略"路线：报告 §6。

> 本模型文档为实施规范（implementation spec）；调研结论以报告为准，二者符号以本文 §2 映射表为桥接。
