# 汽车 Skyhook（天棚阻尼）模型研究综述

## 摘要

Skyhook（天棚阻尼）是半主动悬架控制中历史最悠久、产业落地最广的控制思想，由 Karnopp、Crosby 与 Harwood 于 1974 年在 Lord Corporation 提出并发表于 ASME《Journal of Engineering for Industry》[cite:2][cite:1]。其核心思想是：在簧载质量（车身）与一个假想的"惯性天空"之间虚拟安装一个阻尼器，从而抑制车身的绝对垂向运动，而非仅衰减车身—车轮间的相对运动。由于真实阻尼器只能安装在车身与车轮之间，工程上通过在线调节（切换）阻尼系数来"逼近"理想天棚力，构成半主动（semi-active）控制——只调节阻尼、不注入机械能量，功耗和成本远低于全主动悬架，性能却接近全主动[cite:2][cite:1]。

本报告梳理 Skyhook 的数学模型（四分之一车模型、理想天棚律、ON-OFF 与连续两类半主动实现）、性能特征与局限（簧载共振抑制显著、轮跳频段与轮胎动载荷无改善）、其家族变体（Groundhook、Hybrid/Skygroundhook、ADD、SH-ADD、双天棚、VISKY 虚拟惯量天棚等）、产业落地（CDC 与磁流变 MagneRide 系统均以 Skyhook 类算法为基础）、以及研究前沿（强化学习与可微物理/控制协同设计）。结尾给出从可微物理（differentiable physics）视角对后续工作的启示。

---

## 1. 背景与概念界定

### 1.1 悬架分类与半主动概念

按控制方式，悬架分为被动（passive）、半主动（semi-active）与主动（active）三类[cite:1]。被动悬架参数固定，无法在各种路况与载荷下灵活适应，难以调和平顺性与操稳性的固有矛盾；半主动与主动悬架可按工况在线调节阻尼（或刚度），成为悬架技术的主要发展方向[cite:1]。悬架控制策略大体可分三类：基于车辆状态判断的规则类（以 Skyhook 为代表）、基于控制理论类（PID、LQR、SMC、MPC 等）、以及新兴的现代智能类（神经网络、预瞄控制等）；目前工程中量产落地的仍以前两类及其改进型为主[cite:1]。

半主动方案的定义源自 Karnopp 等人 1974 年论文：一类能够响应振动系统一般反馈信号来控制振动、但**不需要伺服机构功率供给**的力发生器；仿真研究表明其性能可与全主动系统相当[cite:2]。换言之，半主动阻尼器只能"耗能"（阻尼力始终与相对速度反向），其可调度量是阻尼系数 c ∈ [c_min, c_max]，而非任意力输出——这一"无源性（passivity）约束"是所有 Skyhook 类算法设计的出发点。

### 1.2 Skyhook 的起源

Skyhook 阻尼策略由 Karnopp、Crosby、Harwood 于 1974 年正式提出（更早有 1973 年 ASME 会议版本 73-DET-122），作者单位为 Lord Corporation（Erie, PA）——一家后来在磁流变（MR）流体与阻尼器领域同样领先的减振器厂商[cite:2][cite:3]。理想 Skyhook 控制在"天空中的固定点"与车身之间虚拟安装一个阻尼器，抑制车身垂向运动，使车身更平稳、改善平顺性与行驶平顺度[cite:1]。这一思想至今仍是乘用车平顺性控制的行业基线[cite:4]。

---

## 2. 基础模型与控制律

### 2.1 四分之一车模型（Quarter-Car Model）

Skyhook 类算法的分析普遍基于四分之一车模型：簧载质量 m_s（车身）经弹簧 k_s 与被动阻尼 c_s 连接非簧载质量 m_u（车轮总成），轮胎简化为弹簧 k_t；z_s、z_u、z_r 分别为车身、车轮、路面高程。其运动方程为[cite:3]：

$$
m_s \ddot{z}_s = -k_s(z_s - z_u) - c_s(\dot{z}_s - \dot{z}_u) + F
$$

$$
m_u \ddot{z}_u = k_s(z_s - z_u) + c_s(\dot{z}_s - \dot{z}_u) - k_t(z_u - z_r) - F
$$

其中 F 为半主动阻尼器提供的可调阻尼力（主动力为零）。该 2 自由度系统存在两个典型模态：簧载（车身）共振约 1–2 Hz，轮跳（wheel hop）共振约 10–15 Hz[cite:4]。

### 2.2 理想 Skyhook 与半主动可实现性

**理想天棚**：若在车身与惯性天空之间虚拟安装阻尼系数为 c_sky 的阻尼器，其对车身的理想作用力为

$$
F_{\mathrm{ideal}} = -c_{\mathrm{sky}} \dot{z}_s
$$

即阻尼力始终对抗车身的**绝对**速度。该力与车轮运动无关，理论上可对车身形成"接地"式绝对衰减，不受路面激励直接冲击。但真实阻尼器只能安装在车身与车轮之间，其出力必然正比于**相对**速度 $\dot{z}_{\mathrm{rel}} = \dot{z}_s - \dot{z}_u$ 并受无源性约束（力只能抵抗相对运动）。因此理想天棚力一般不可直接实现，需要一个"映射"策略：在阻尼器能力范围 [c_min, c_max] 内，尽量复现理想天棚力的效果[cite:3][cite:4][cite:1]。

### 2.3 ON-OFF 型（两点式）半主动控制律

最经典的实现是 Karnopp 的开关型策略：当阻尼器的相对运动方向恰好能对抗车身绝对运动时（$\dot{z}_s \cdot \dot{z}_{\mathrm{rel}} > 0$），选择最大阻尼；否则选择最小阻尼（或零阻尼），让弹簧自然工作[cite:5][cite:3]：

$$
c_f = \begin{cases} c_{\max}, & \dot{z}_s (\dot{z}_s - \dot{z}_u) > 0 \\ c_{\min}\ (\text{或}\ 0), & \text{其他} \end{cases}
$$

等价地用力表达（VISKY 综述中的形式）[cite:3]：

$$
F_{\mathrm{sky}} = \begin{cases} -C_{\mathrm{sky}} \dot{z}_{\mathrm{rel}}, & \dot{z}_s \dot{z}_{\mathrm{rel}} > 0 \\ 0, & \dot{z}_s \dot{z}_{\mathrm{rel}} \le 0 \end{cases}
$$

直观判据（产业博客的工况解释）：车身向上而车轮向下（相反）时开大阻尼拉住车身；车轮比车身下落更快时开小阻尼，避免把车身"拽"向路面[cite:4]。ON-OFF 策略对单自由度隔振系统而言已被证明是接近最优的开关控制形式，其推广与优化在后续文献中被系统研究[cite:6]。

### 2.4 连续型（continuous）半主动控制律

连续型策略不进行两点切换，而是让等效阻尼系数尽量逼近理想天棚所需的比值，并受限于 [c_min, c_max]：

$$
c_{\mathrm{eq}} = \mathrm{clamp}\!\left(\frac{c_{\mathrm{sky}} \dot{z}_s}{\dot{z}_{\mathrm{rel}}},\ c_{\min},\ c_{\max}\right), \qquad F_d = -c_{\mathrm{eq}} \dot{z}_{\mathrm{rel}}
$$

即：在无源性允许的范围内，逐时刻选择"最接近理想天棚力"的实际阻尼[cite:7][cite:3]。连续型避免了开关切换引入的力突变与高频激励，代价是需要阻尼系数连续可调（比例阀 CDC / 磁流变阻尼器）[cite:3]。

> 注：各文献符号约定（z 与位移正方向、F 的正负）略有差异，以上以 VISKY 论文的约定为准[cite:3]；结论不依赖符号约定。

---

## 3. 性能特征与固有局限

### 3.1 频域特征

- **目标频段**：Skyhook 针对 1–2 Hz 的车身颠簸（body bounce）模态最有效；Groundhook 类策略则针对 10–15 Hz 的轮跳（wheel hop）模态[cite:4]。
- **簧载共振**：Skyhook 能显著衰减簧载质量共振峰，提升平顺性[cite:8]。
- **非簧载共振**：Skyhook 不处理车轮振动，**无法降低非簧载质量对应的共振峰**，对操纵稳定性与接地性改善有限[cite:8]。
- **中高频发颠（harshness）**：c_sky 取值过大时，在 5–10 Hz 频段反而会放大传入车身的振动，形成"发麻"的高频冲击感[cite:4]。

### 3.2 量化改善幅度

不同研究给出的改善数字差异较大，取决于路面等级、车速、阻尼器可调范围与评价指标：

- 产业资料称 Skyhook 可使车身加速度 RMS 相对被动悬架降低 15%–25%（对比表）乃至 30% 以上（正文表述），且整个系统功耗仅约 50 W，可在 16/32 位车规 MCU 上运行[cite:4]；
- 针对 A 级路面、低车身高度、高速工况（平顺性权重较高）的对比：SH-ADD 相对被动悬架簧载加速度 RMS 降低 5.82%，自适应改进天棚降低 14.77%[cite:9]；
- 2025 年综述则指出传统 Skyhook 类算法虽然计算简单、易于工程实现，但**平顺性改善有限**，由此催生了大量改进型算法[cite:1]。

综合判断：Skyhook 在簧载共振频段的改善稳定可复现（两位数百分比量级），但全频段综合改善取决于调参与工况，文献间存在 5%–30% 的显著分散[cite:1][cite:4][cite:9]。

### 3.3 固有局限汇总

1. **忽视轮胎接地性**：Skyhook 只关心车身平稳，不管理轮胎动载荷，快速变线等工况下动态接地力波动可能损害抓地力[cite:4][cite:8]。
2. **频段窄**：标准混合策略往往只对窄频段有效，偏离名义频段时透振更高[cite:3]。
3. **切换不连续**：两点式（及天棚/地棚间切换的混合策略）存在力的突变，引入高频振动成分，可能与高阶模态共振[cite:3]。
4. **状态获取**：需要簧载绝对速度 $\dot{z}_s$，而工程中直接可用的是加速度信号，速度需经积分/滤波获得，存在漂移与相位问题——这正是后续 ADD 类算法（改用加速度）与各种观测器方案的动机[cite:10]。
5. **对非簧载质量的附加质量不友好**：轮毂电机等增大簧下质量的场合，常规天棚策略难以保证接地性，需要双天棚等改进[cite:11]。

---

## 4. Skyhook 家族变体

### 4.1 Groundhook（地棚）

Groundhook 将虚拟阻尼器改为连接**非簧载质量与地面**，目标是压制车轮的绝对垂向跳动、稳定轮胎动载荷，服务于操稳与安全[cite:4][cite:12]。其控制逻辑与天棚对称（以车轮绝对速度 $\dot{z}_u$ 替代 $\dot{z}_s$）[cite:3]：

$$
F_{\mathrm{gr}} = \begin{cases} -C_{\mathrm{gr}} \dot{z}_{\mathrm{rel}}, & \dot{z}_u \dot{z}_{\mathrm{rel}} < 0 \\ 0, & \text{其他} \end{cases}
$$

Groundhook 单独使用会使车内乘员感受到极端生硬的振动（路感几乎直通车身），因此几乎总是与天棚混合使用[cite:4]。该思想在文献中被归于 Novak 与 Valášek 提出的"附加接地惯量阻尼器"概念[cite:12]。

### 4.2 Hybrid / Skygroundhook（加权混合）

由于平顺性与接地性天然冲突，工程上采用加权混合[cite:4]：

$$
F_{\mathrm{hybrid}} = \alpha \, F_{\mathrm{sky}} + (1-\alpha) \, F_{\mathrm{gr}}
$$

α 实时调度：高速巡航 α→0.8 偏舒适；激烈驾驶 α→0.2 偏抓地[cite:4]。学术上，Gonçalves 与 Ahmadian 系统研究了这种"hybrid control policy"，证明其能同时兼顾两类指标[cite:13]；Díaz 等人将天棚与地棚融合设计，为最优悬架控制提供了新视角[cite:1]。VISKY 论文给出了一种连续 Skygroundhook 的紧凑表达：当 $\dot{z}_s \dot{z}_{\mathrm{rel}} > 0$ 时出力 $-C_{\mathrm{sky}} \dot{z}_s - C_{\mathrm{gr}} \dot{z}_u$，否则回落到被动阻尼，再将期望力通过 clamp 映射为等效阻尼系数[cite:3]：

$$
F_{\mathrm{sg}} = \begin{cases} -C_{\mathrm{sky}} \dot{z}_s - C_{\mathrm{gr}} \dot{z}_u, & \dot{z}_s \dot{z}_{\mathrm{rel}} > 0 \\ -C_{\mathrm{passive}} \dot{z}_{\mathrm{rel}}, & \text{其他} \end{cases}, \qquad
C_{\mathrm{eq}} = \mathrm{clamp}\!\left(-\frac{F_{\mathrm{sg}}}{\dot{z}_{\mathrm{rel}}},\ 0,\ C_{\max}\right)
$$

### 4.3 ADD 与 SH-ADD（Savaresi 学派）

- **ADD（Acceleration-Driven Damper，加速度驱动阻尼）**：Savaresi 等人 2005 年提出的舒适导向最优半主动算法。其控制逻辑与天棚平行，但用簧载**加速度**替代簧载速度，与相对速度比较——加速度恰是传感器易测量、也是平顺性指标直接关联的量。ADD 在高频段优于天棚，低频段（簧载共振附近）不及天棚[cite:10][cite:14]。
- **SH-ADD（混合天棚-ADD）**：针对 SH 与 ADD 各自"半频段最优"的局限，Savaresi 与 Spelta 将两者组合为两点式混合策略，在全频段内降低车身振动加速度[cite:15][cite:9]。其控制律可写成（z 为簧载位移，$\alpha$ 为混合参数，$\dot z_t$ 为车轮速度）[cite:16]：

$$
c_{\mathrm{in}} = \begin{cases} c_{\min}, & \left(\ddot{z}^{2} - \alpha^{2} \dot{z}^{2} \le 0\right) \land \left(\dot{z}(\dot{z} - \dot{z}_{t}) \le 0\right) \\ c_{\max}, & \text{其他} \end{cases}
$$

SH-ADD 只需簧载加速度与悬架相对速度（可由位移传感器差分获得），无需绝对速度观测，是工程上非常实用的"免调参"基线，后续 Martellosio 等人还提出了时变参数版本以适应不同路面[cite:1][cite:15]。

### 4.4 双天棚与其他改进

- **双天棚（double sky-hook）**：针对轮毂电机簧下质量增大的问题，组合"簧载运动阻尼系数"与"非簧载运动阻尼系数"两个天棚项，以改善接地性（road-holding）[cite:11]。
- **改进天棚 / 增广天棚**：在理想天棚力上叠加悬架相对速度项，兼顾非簧载质量响应[cite:5]。
- **增益调度与硬件在环（HIL）调参**：ASME 2002 年工作提出改进天棚模型、按工况调度增益，并通过 HIL 完成参数整定[cite:12]。
- **路面自适应**：北京理工大学团队用 LSTM 识别路面不平度等级，结合改进天棚与模糊控制，按悬架响应在线调整车身高度与阻尼，在多工况下显著优于被动与 SH-ADD（前述 14.77% 数据即来自该系统）[cite:9]。

### 4.5 VISKY：虚拟惯量天棚（2025/2026 前沿）

最新工作 VISKY（Virtual Inertia Skyhook）把 Skyhook 家族统一为一个四参数广义力律，并在连续天棚-地棚基线上叠加簧载与非簧载**加速度**反馈[cite:3]：

$$
F_d = -P_{\mathrm{sky}} \dot{z}_s - D_{\mathrm{sky}} \ddot{z}_s - P_{\mathrm{gr}} \dot{z}_u - D_{\mathrm{gr}} \ddot{z}_u
$$

其中 P 项单位 N·s/m（阻尼型），D 项单位 kg（惯量型）。令不同参数归零即可退化为：纯天棚（P_gr=D_gr=0）、纯地棚（P_sky=D_sky=0）、连续天棚-地棚（D_sky=D_gr=0）[cite:3]。闭环方程中，加速度反馈项并不改变物理惯量，而是构成一个"虚拟惯量矩阵"（质量样矩阵），在**轮跳频段附近**显著改善车身与车轮加速度指标，同时保持代数运算的低计算负担，适配算力受限的车载/机器人平台[cite:3]。

### 4.6 家族对比一览

| 方法 | 反馈信号 | 目标 | 频段倾向 | 实现难度 | 代表出处 |
|---|---|---|---|---|---|
| 理想天棚（主动） | 簧载绝对速度 | 车身平稳 | 1–2 Hz 最优 | 需主动作动器 | [cite:2] |
| ON-OFF 天棚 | 簧载速度、相对速度 | 平顺性 | 簧载共振 | 极低（两点阻尼） | [cite:2][cite:5] |
| 连续天棚 | 同上 | 平顺性 | 簧载共振 | 需连续可调阻尼 | [cite:7][cite:3] |
| Groundhook | 车轮速度、相对速度 | 接地性 | 10–15 Hz | 低 | [cite:12][cite:4] |
| Hybrid/Skygroundhook | 两者加权 | 平顺+接地 | 可调度 | 低-中 | [cite:13][cite:4] |
| ADD | 簧载加速度 | 平顺性 | 高频段最优 | 低 | [cite:14] |
| SH-ADD | 加速度+速度 | 平顺性 | 全频段 | 低 | [cite:15][cite:16] |
| 双天棚 | 双侧速度 | 接地性（轮毂电机） | 轮跳 | 中 | [cite:11] |
| VISKY | 双侧速度+加速度 | 平顺+接地（虚拟惯量） | 全频段，轮跳改善明显 | 中（需加速度信号+MR 逆映射） | [cite:3] |

---

## 5. 执行器与产业落地

### 5.1 可调阻尼硬件

Skyhook 是"算法"，其物理载体是可调阻尼减振器，当前两类主流[cite:4][cite:17]：

1. **CDC（Continuous Damping Control）**：电液比例阀控制油液流阻，实现阻尼系数连续可调（如 ZF Sachs / Tenneco / BWI 的量产产品）。
2. **MR 磁流变阻尼器**：载液中含微米级磁性颗粒，外加磁场使颗粒成链、屈服应力在毫秒级内改变等效阻尼（BWI MagneRide 等）。MR 阻尼器常用的数学模型是 Bouc-Wen 滞回模型[cite:3]：

$$
F_{\mathrm{MR}} = (C_{0a} + C_{0b} u)\,\dot{z}_{\mathrm{rel}} + k_0 z_{\mathrm{rel}} + (\alpha_a + \alpha_b u)\,\xi
$$

其中 u 为电压指令、ξ 为滞回状态。常见模型的标定与辨识方法综述见 2024 年 MR 半主动悬架综述[cite:17]。

### 5.2 量产应用：Skyhook 是产业基线

- **MagneRide（磁流变）**：其半主动悬架算法即基于 Skyhook Damping Control Law，状态变量取 7 个整车速度量（车身垂荡、侧倾等各速度分量），在 Corvette、Cadillac 等车型量产[cite:18]。
- **CDC（连续阻尼控制）**：产业媒体明确描述 CDC 采用"Skyhook 策略"——想象把车辆悬挂在空中的一根虚拟轨道上，控制模块实时提供适当阻尼水平以保证最平顺的行驶；激烈驾驶时则在 ESC 介入前就让车辆保持稳定路线[cite:19]。
- **覆盖面**：产业估计 Skyhook 与 Hybrid 类算法运行于 95% 以上配备可调阻尼减振器的量产乘用车上（CDC 或 Magnetic Ride），原因即在于其算力需求低、传感器成本低（普通加速度计即可）、效果直接[cite:4]。**（注：此数字为单一产业来源的行业估计，未见学术统计交叉验证）**
- 出身线索：1974 年 Skyhook 原始论文出自 Lord Corporation，而 Lord 正是磁流变流体技术的先驱——Skyhook 算法与 MR 阻尼器存在深厚的历史渊源[cite:2]。

### 5.3 摩托车与轨道车辆

- **摩托车**：Öhlins Smart EC 等电子悬架（半主动阻尼）已量产搭载于 Ducati 等高性能摩托车，原理与汽车半主动天棚一致[cite:20]。
- **轨道车辆**：半主动 MR 阻尼器与天棚类算法同样用于高速列车横向半主动悬挂，以抑制车体横向振动；2026 年 Mechanics & Industry 的工作研究了基于改进 MR 阻尼器的列车半主动协调控制[cite:21]。

---

## 6. 研究前沿：学习类方法与可微设计

### 6.1 强化学习半主动控制

近年大量工作把悬架控制视为序贯决策问题，用深度强化学习（DRL）直接学习阻尼调度策略，并逐步走向实车：

- 2024 年发表于 Applied Sciences 的工作完成了"建模—训练—整车仿真验证—实车部署"全流程的 RL 半主动垂向动力学控制器，实车测试改善平顺性[cite:22]。
- Physics-Guided RL 框架将 DRL 策略网络输出显式约束为物理可实现的"主动刚度 k_a 与主动阻尼 c_a"，保证控制动作物理可行[cite:23]。
- Lee（2023）用深度 RL 学习半主动悬架控制策略时，即以混合 SH-ADD 为对照基线，说明 SH 类算法是学习类方法的通用 benchmark[cite:16]。

### 6.2 可微物理与"硬件即策略"（Hardware-as-Policy）

与本项目（DiffPhysSkyHook）最相关的是**控制-结构协同设计（Control Co-Design, CCD）**方向：把悬架物理参数本身作为可微分量纳入优化图，与控制策略联合求梯度优化：

- Digital Twin + DRL 全车主动悬架协同设计（arXiv:2512.03891）：在 PPO 更新中同时优化四角悬架的刚度 k_s、阻尼 c_s 与策略网络参数，使硬件与策略在同一学习回路中共同进化[cite:24]。
- "硬件即策略"思想（2026, Structural and Multidisciplinary Optimization）：将物理参数表示为计算图中的可微组件，使基于梯度的 DRL 算法（PPO/TRPO）可对机械设计变量直接求导，实现数字孪生驱动的全车悬架控制协同设计[cite:25]。
- 多代 CCD 框架（arXiv:2510.10694）：DRL + 自动微分协同优化机械参数与控制策略，并用量分回归从实测道路数据中学习环境不确定性，二轮优化后动态响应更优、控制轨迹更平滑[cite:26]。

### 6.3 Skyhook 在学习时代的角色

Skyhook/SH-ADD 由于计算便宜、物理透明、免复杂调参，在学习类研究中承担三个角色：(1) 对照基线（学习策略必须证明显著优于 SH 类才具落地价值）；(2) 策略先验/结构约束（把 RL 输出限定在天棚家族参数内，如 VISKY 式广义四参数）；(3) 奖励塑形的安全回退层（学习失败时退化为 SH）[cite:16][cite:3][cite:22]。

---

## 7. 结论

1. Skyhook 是半主动悬架的奠基性控制范式（Karnopp et al., 1974），本质是"用可调阻尼器逼近理想惯性参考阻尼"，在簧载共振频段（1–2 Hz）稳定改善平顺性两位数百分比量级，功耗约数十瓦，是量产可调阻尼车的默认算法基线[cite:1][cite:2][cite:4]。
2. 其固有局限同样清晰：不改善轮跳频段（10–15 Hz）与轮胎动载荷，两点式切换引入高频力突变，依赖绝对速度信号——由此发展出 Groundhook、Hybrid/Skygroundhook、ADD、SH-ADD、双天棚直至 VISKY 的完整谱系[cite:3][cite:8][cite:15]。
3. 产业上，CDC 与 MagneRide 等系统均以天棚类算法为核心；学术上，研究重心已从"设计更好的规则"转向"学习更好的策略"，且趋势是把物理参数与控制策略放进同一个可微计算图做协同设计——天棚族算法在其中充当基线与结构先验[cite:18][cite:19][cite:24][cite:25]。

---

## 8. 对后续工作的启示（可微物理视角）

以下为基于上述调研的综合分析，供后续建模实现参考：

### 8.1 可微性分析：选择"平滑"的 Skyhook 变体

- ON-OFF 天棚的开关条件 $\dot z_s \dot z_{\mathrm{rel}} > 0$ 在切换边界不可导，且两点阻尼导致梯度几乎处处为零（对设计参数而言）——**不适合直接放进可微仿真做端到端优化**；
- 连续天棚的 clamp（饱和）函数是次可微的，梯度可在未饱和区正常回传；
- VISKY 广义律（速度+加速度线性反馈）是完全光滑的代数力律，且统一了天棚/地棚/混合为四个可微参数 (P_sky, D_sky, P_gr, D_gr)——**最适合作可微物理中的参数化控制器**，四个参数本身就是天然的"可微设计变量"[cite:3]。

### 8.2 建议的技术路线

1. **模型**：从四分之一车模型起步（式 §2.1），轮胎线性弹簧 k_t；需要滞回与非线形性时引入 Bouc-Wen MR 模型[cite:3]。
2. **基线阶梯**：被动 → ON-OFF SH → 连续 SH → SH-ADD → Skygroundhook/VISKY，再叠加学习类策略对照[cite:3][cite:15]。
3. **设计变量**：可把 (c_min, c_max, c_sky, α) 或 VISKY 四参数，连同 k_s、c_s 一起作为可微设计变量，采用"硬件即策略"方式与控制策略联合优化[cite:24][cite:25]。
4. **评价指标三件套**：簧载加速度 RMS（平顺性）、悬架动行程（包络/限位）、轮胎动载荷（接地性）——三者构成经典的不可能三角，任何单一目标的极端优化都会恶化其余两项[cite:4][cite:8]。
5. **路面输入**：用 ISO 8608 分级随机路面 + 半正弦凸块 + 正弦扫频三类激励做验证，与 VISKY 论文一致，便于横向对比[cite:3]。

---

## Sources

- [cite:1] Wang, J. et al., "Skyhook-Based Techniques for Vehicle Suspension Control: A Review of the State of the Art", Machines 13(8):727, MDPI, 2025. https://www.mdpi.com/2075-1702/13/8/727
- [cite:2] Karnopp, D., Crosby, M.J., Harwood, R.A., "Vibration Control Using Semi-Active Force Generators", Journal of Engineering for Industry, 96(2):619–626, 1974. DOI 10.1115/1.3438373（原文 PDF：https://sci-hub.cat/storage/zero/7304/59da6b408f1d7afb7b7334ae314195a0/karnopp1974.pdf）
- [cite:3] Lim, H., Lee, J.W., Choi, S.-B., Choi, J.B., "VISKY: Virtual Inertia Skyhook Control for Semi-Active Suspension Systems Using Magnetorheological Dampers", arXiv:2503.12891v2, 2026. https://arxiv.org/html/2503.12891
- [cite:4] Liu, J., "Skyhook, Groundhook, and H∞ Control Algorithms in Semi-Active Suspension", Dowway Vehicle Technology Article, 2026. https://www.dowwayvehicle.com/semi-active-suspension-skyhook-groundhook-h-infinity/
- [cite:5] 《考虑能控特性的天棚阻尼半主动控制算法性能分析》，《振动与冲击》（上海交通大学）. https://nvc.sjtu.edu.cn/CN/article/downloadArticleFile.do?attachType=PDF&id=2453
- [cite:6] "Generalisation and optimisation of semi-active, on-off switching controllers for single degree-of-freedom systems", Journal of Sound and Vibration (White Rose eprints). https://eprints.whiterose.ac.uk/id/eprint/80980/1/jsv_optimal_draft.pdf
- [cite:7] "Study and Review on Vehicle Suspension Control Theories and Introduction of Novel Adaptive Skyhook Control System", AJBAS. http://ajbasweb.com/old/ajbas/2015/Special%20IPN%20KRABI%20Sept/1-12.pdf
- [cite:8] "Development of a systematic and practical methodology for the design of vehicles semi-active suspension control system", Vehicle System Dynamics, Taylor & Francis. https://www.tandfonline.com/doi/10.1080/00423110902971809
- [cite:9] 《基于路面识别的半主动空气悬架控制》，北京理工大学学报（2025）. https://journal.bit.edu.cn/zr/article/doi/10.15918/j.tbit1001-0645.2024.193
- [cite:10] "Advancements in Semi-Active Automotive Suspension Systems with Magnetorheological Dampers: A Review", Applied Sciences 14(17):7866, MDPI, 2024. https://www.mdpi.com/2076-3417/14/17/7866
- [cite:11] "A Double Sky-Hook Algorithm for Improving Road-Holding Property in Semi-Active Suspension Systems for Application to In-Wheel Motor", Applied Sciences 11:8912, MDPI. https://mdpi-res.com/d_attachment/applsci/applsci-11-08912/article_deploy/applsci-11-08912-v3.pdf
- [cite:12] "Modified Skyhook Control of Semi-Active Suspensions: A New Model, Gain Scheduling, and Hardware-in-the-Loop Tuning", ASME Journal, v124, 2002. https://cogno.pusan.ac.kr/sites/cogno/download/eng/1_35_ASME_v124_2002_H.pdf
- [cite:13] Gonçalves, F.D., Ahmadian, M., "A hybrid control policy for semi-active vehicle suspensions", 2002. https://www.sci-hub.ru/download/2024/6954/507362f48bd4b48b87a5f88566b5290c/goncalves2002.pdf
- [cite:14] Savaresi, S.M., Silani, E., Bittanti, S., "Acceleration-Driven Damper (ADD): An Optimal Control Algorithm for Comfort-Oriented Semiactive Suspensions", Journal of Dynamic Systems, Measurement, and Control, 127(2):218–229, 2005（转引自 Mechanics & Industry 参考文献）. https://www.mechanics-industry.org/articles/meca/full_html/2026/01/mi20250123/mi20250123.html
- [cite:15] "Study on Sliding Mode Variable Structure Control Strategy of Semi-Active Suspension Based on Power Flow Analysis", Preprints.org, 2026. https://www.preprints.org/manuscript/202608.0713
- [cite:16] Lee, "Deep Reinforcement Learning of Semi-active Suspension Controller for Vehicle Ride Comfort", 2023. https://sci-hub.cat/storage/2024/9043/96a74d3be856124492b723be553477dc6645f7547ea6d63bdc89df3092af0e10/lee2023.pdf
- [cite:17] 同 [cite:10]（MR 阻尼器建模与辨识综述章节）.
- [cite:18] "Concurrent Research and Development of a Magnetic Ride Control System"（MagneRide 半主动算法基于 Skyhook Damping Control Law）. https://www.sci-hub.su/storage/moscow/3125/1c6c7ab9b2e182febfae6ab123157648/concurrent-research-and-development-of-a-magnetic-ride-control-s.pdf
- [cite:19] "CDC employs the 'Skyhook' strategy", WardsAuto. https://www.wardsauto.com/news/archive-wards-big-leagues/766883/
- [cite:20] Öhlins, "Our History and Development Over 50 Years"（Smart EC 电子控制悬架）. https://www.ohlins.com/en-us/about-us/history
- [cite:21] "Research on semiactive coordinated control of trains on the basis of an improved model of magnetorheological dampers", Mechanics & Industry, 2026. https://www.mechanics-industry.org/articles/meca/full_html/2026/01/mi20250123/mi20250123.html
- [cite:22] "Reinforcement Learning for Semi-Active Vertical Dynamics Control with Real-World Tests", Applied Sciences 14(16):7066, MDPI, 2024. https://www.mdpi.com/2076-3417/14/16/7066
- [cite:23] "Physics-Guided Reinforcement Learning System for Realistic Vehicle Active Suspension Control", arXiv:2408.08425, 2024. https://arxiv.org/html/2408.08425v1
- [cite:24] "Digital Twin-based Control Co-Design of Full Vehicle Active Suspensions via Deep Reinforcement Learning", arXiv:2512.03891, 2025. https://arxiv.org/html/2512.03891v1
- [cite:25] "Reinforcement learning-based control co-design of digital twin-enabled full-vehicle active suspension systems", Structural and Multidisciplinary Optimization, 2026（hardware-as-policy：物理参数作为可微组件）. https://yiping514.github.io/files/2026_SMO_DTRL.pdf
- [cite:26] "Digital Twin-enabled Multi-generation Control Co-Design with Deep Reinforcement Learning", arXiv:2510.10694, 2025. https://arxiv.org/html/2510.10694v1/
