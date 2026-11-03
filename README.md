# DiffPhysSkyHook

基于**可微物理仿真**（[Quadrants](https://pypi.org/project/quadrants/) 可微内核 + torch autograd 桥）的半主动悬架（Skyhook）建模、控制与端到端策略训练研究项目。

核心思路：把 1 kHz 四分之一车物理内核的 rollout 包进 `torch.autograd.Function`，使复合均方目标 **J_sim**（车身加速度 + 悬架行程 + 轮胎力 + 行程包络铰链）对阻尼力指令**可反传**，从而用策略梯度端到端训练阻尼力规划网络（10 Hz 指令 / 1 kHz 执行）。

## 主要成果

| 结果 | 数值 |
|---|---|
| 策略 vs 连续天棚（留出种子 777，模式 1：仅历史观测） | j_sim **4.513，−14.8%** |
| 策略 vs 连续天棚（模式 2：另加前向几何事件预览） | j_sim **4.111，−20.9%** |
| 特权全窗规划器上界（非因果） | 2.988 / 2.958 |
| 测试规模 | **116 passed**（Plan A/B 67 回归门 + Plan C/D 49 新增） |
| 数值精度 | 三路梯度互验 ≤ 2.7e−9；CPU/GPU 对拍 ≤ 1e−12；饱和槽梯度精确为 0 |

模式归属（T28 负对照协议）：预览通道的净收益 ~6 个百分点，严格归因于「提前看见坑洞/凸起」；粗糙度信息按感知分工铁律不可见（settling 负对照 0.514 vs 0.514）。

## 环境安装

```bash
conda env create -f environment.yml   # env: diffphy (python 3.12, quadrants 1.3.0, torch)
conda activate diffphy
```

## 快速上手

所有命令从**项目根目录**执行（tests 依赖根目录 `conftest.py`）。

### 运行全部测试

```bash
python -m pytest tests/ -q                    # 116 用例，~13 min（含两次 400 迭代 fixture 训练）
python -m pytest tests/test_f0_discrete.py -q # 只跑某一阶段
```

GPU 用例（`test_f4_gpu.py`）需要真实 CUDA 设备，在无 GPU 环境会自动跳过。

### 训练策略网络

```bash
python -c "
from train_policy import TrainConfig, train
cfg = TrainConfig(mode=2, B=8, H=100, t_w=1.0, n_iter=400,
                  fixed_road_seed=None)   # 域随机化（发现 D-1：固定夹具必过拟合）
train(cfg)                                # checkpoints/ 下每 50 迭代 + best 落盘
"
```

关键配置（`TrainConfig`）：`mode`（1/2 感知模式）、`B`（批量）、`n_iter`、`w_jerk`（默认 1e-4，Plan D G2 Pareto 标定）、`decay`（段内伴随衰减）、`device`（`cuda` 可用但标定显示 GTX 1060 上慢 1.8×）。

训练内建**三种子分离**：训练种子 0 / 选型种子 555（每 25 迭代 select 评测 + best checkpoint）/ 评测种子 777（禁止进入任何训练决策）。

### 留出评测与基线对比

```bash
python eval_policy.py                 # 默认逐 episode 域随机化
python eval_policy.py --fixed-roads   # 固定夹具模式（仅用于过拟合复现实验）
```

输出六指标表：j_sim、J_acc、J_travel、J_tire、jerk、ms_ft，控制器覆盖被动 / 连续天棚 / ON-OFF 天棚 / 两模式策略 / 特权 oracle。

## 目录结构

```
skyhook/            物理内核与训练组件
  model.py/sim.py       Plan A：解析天棚模型（model2/sim2 为指令通道前身）
  model2b.py/sim2b.py   Plan B：外力指令通道（批量 + 反传伴随）
  model3.py/sim3.py     Plan C：序列批量内核（B_MAX=64, T_B_MAX=4096）+ J_sim
  road.py/road_seq.py   路面：解析族 / 离散 AR 切换 + 带符号事件族
  bridge.py             torch autograd 桥（前向宿主 setup，反向纯 numpy 伴随）
  policy.py             PolicyMLP（59,716 参数）+ 观测装配
  params.py             半主动悬架参数与 clip 语义
  _runtime.py           quadrants 后端绑定（ensure_init）
train_policy.py     分段训练回路 + select 协议 + cosine 调度
eval_policy.py      闭环滚动器、基线工厂、特权 oracle、六指标评测
tests/              T1–T31 测试矩阵（f0–f4 / g1–g3 按 Plan 阶段组织）
docs/               各阶段方案文档（权威）、参数参考、训练测试报告
checkpoints/        训练产物（不入库，可由 train_policy 复现）
```

## 文档索引

| 文档 | 内容 |
|---|---|
| [docs/plan_a_skyhook_model.md](docs/plan_a_skyhook_model.md) | Plan A：天棚模型与可微内核 |
| [docs/plan_b_force_input_dev_test_plan.md](docs/plan_b_force_input_dev_test_plan.md) | Plan B：外力指令通道与批量反传 |
| [docs/plan_c_policy_training_dev_test_plan.md](docs/plan_c_policy_training_dev_test_plan.md) | Plan C：策略训练（含 §9 执行记录与全部发现） |
| [docs/plan_c_training_test_report.md](docs/plan_c_training_test_report.md) | Plan C 训练测试报告 |
| [docs/plan_d_training_optimization_dev_test_plan.md](docs/plan_d_training_optimization_dev_test_plan.md) | Plan D：训练优化（G1 选型 / G2 正则 / G3 扩容，进行中） |
| [docs/parameters_reference.md](docs/parameters_reference.md) | 悬架参数与 J_sim 权重标定 |

## 工程纪律（要点）

- **单变量推进**：每阶段只改一个变量，测试全绿才进下一阶段；Plan A/B 67 用例为永久回归门
- **种子纪律**：训练(0) / 选型(555) / 测试(777) 三分离，777 出现在实现代码即 Gate 失败
- **公平对照**：两模式策略同参数量、同种子、同分布训练，归因靠 T28 负对照
- **域随机化**：路面逐 episode 重采样；环境进分布、设备不进、梯度不穿内核

## 当前状态

- ✅ Plan A（天棚模型）、Plan B（力指令通道）、Plan C（策略训练，108 tests）
- ✅ Plan D G1（选型协议）/ G2（w_jerk=1e-4 采纳）
- 🔄 Plan D G3（B=32/800 迭代扩容 + decay 消融，训练运行中）
- ⏳ Plan D G4（oracle 收敛质量）/ G5（收官）
