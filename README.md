# 基于方位的多智能体编队控制复现

本仓库复现 Zhao 和 Zelazo 论文 *Translational and Scaling Formation Maneuver Control via a Bearing-Based Approach* 中的核心算法。代码使用 Python 实现基于方位拉普拉斯矩阵的 leader-follower 编队控制，并给出静止、恒定速度和时变速度三类 leader 运动的仿真、误差指标和 GIF 动画。

论文原文：[1506.05636v2.pdf](1506.05636v2.pdf)

## 功能

- 构造目标方位、正交投影矩阵和方位拉普拉斯矩阵。
- 通过 $$\lambda_{\min}(\mathcal B_{ff})>0$$ 检查 follower 目标位置是否唯一。
- 实现论文式 (7)：静止或恒定 leader 速度下的相对位置-速度反馈。
- 实现论文式 (10)：时变 leader 速度下的加速度前馈控制。
- 使用四阶 Runge-Kutta 法积分 follower 的双积分器动力学。
- 生成轨迹、全部 agent 速度、总方位误差、PNG 图和 GIF 动画。
- 支持命令行调整 agent 数量、$$k_p$$、$$k_v$$、初始偏差、初速度、随机种子和仿真时长。

## 仓库结构

| 路径 | 用途 |
|---|---|
| `bearing_formation.py` | 核心数学模型、方位拉普拉斯矩阵、控制律、RK4 仿真与指标计算 |
| `animate_formation.py` | 可配置动画脚本，生成编队、所有 agent 速度和方位误差三联 GIF |
| `requirements.txt` | Python 依赖 |
| `results/core_algorithm_reproduction.png` | 论文式 (7) 的 4-agent 静态结果图 |
| `results/formation_to_target.gif` | 8-agent（2 leader + 6 follower）静止 leader 编队形成动画 |
| `results/constant_velocity_demo.gif` | 8-agent 恒定 leader 速度机动动画 |
| `results/time_varying_velocity_demo.gif` | 8-agent 时变 leader 速度机动动画 |
| `results/*_metrics.json` | 与各仿真对应的数值指标 |

## 环境与安装

建议使用 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

主要依赖为 NumPy、Matplotlib 和 Pillow。

## 快速运行

### 1. 核心算法静态图

```bash
python3 bearing_formation.py
```

输出：

- `results/core_algorithm_reproduction.png`
- `results/metrics.json`

### 2. 静止 leader：从初始队形形成目标编队

```bash
python3 animate_formation.py \
  --leader-motion stationary \
  --output results/formation_to_target.gif
```

### 3. 恒定 leader 速度

```bash
python3 animate_formation.py \
  --leader-motion constant \
  --output results/constant_velocity_demo.gif
```

该案例自动使用论文式 (7)。

### 4. 时变 leader 速度

```bash
python3 animate_formation.py \
  --leader-motion time-varying \
  --output results/time_varying_velocity_demo.gif
```

该案例自动使用带加速度前馈的论文式 (10)。

## 常用参数

```bash
python3 animate_formation.py \
  --agents 8 \
  --leader-motion time-varying \
  --kp 1.0 \
  --kv 4.0 \
  --initial-offset 4.0 \
  --initial-speed 0.0 \
  --seed 7 \
  --duration 60 \
  --dt 0.01 \
  --frames 150 \
  --fps 15 \
  --output results/custom_demo.gif
```

| 参数 | 含义 |
|---|---|
| `--agents` | agent 总数，默认为 8，其中前 2 个为 leader |
| `--leader-motion` | `stationary`、`constant` 或 `time-varying` |
| `--kp` | 位置纠偏增益；增大后响应更强，但可能增加振荡 |
| `--kv` | 速度阻尼增益；增大可抑制振荡，过大可能使慢模态收敛变慢 |
| `--initial-offset` | follower 初始位置相对目标的典型偏差幅度 |
| `--initial-speed` | follower 的随机初速度幅度 |
| `--seed` | 随机种子，用于重复初始队形 |
| `--duration` / `--dt` | 仿真时长与 RK4 积分步长 |
| `--frames` / `--fps` | GIF 帧数与播放帧率 |

调大 `--initial-offset` 可以让 follower 运动距离更明显。由于本复现没有主动避碰项，调大初始偏差或初速度后，应检查对应 `*_metrics.json` 中的 `minimum_pairwise_distance`。

## 实验

> 本 README 的控制律编号沿用原论文，即恒速控制律为式 (7)、时变控制律为式 (10)。

### 目标

复现实验的目标是验证：

1. 方位拉普拉斯子矩阵 $$\mathcal B_{ff}$$ 非奇异时，leader 位置能够唯一确定 follower 的目标位置。
2. leader 速度为常值时，论文式 (7) 能使位置和速度跟踪误差收敛。
3. leader 速度时变时，式 (10) 中的加速度前馈能够消除 leader 加速度对跟踪误差的影响。
4. 实际编队在平移和尺度变化过程中仍能保持目标方位。

### 系统模型

每个 follower 采用双积分器动力学：

$$
\dot p_i=v_i,\qquad \dot v_i=u_i,\qquad i\in\mathcal V_f.
$$

对边 $$(i,j)$$，期望方位和对应正交投影矩阵为

$$
g_{ij}^{*}=\frac{p_j^{*}-p_i^{*}}{\|p_j^{*}-p_i^{*}\|},\qquad
P_{g_{ij}^{*}}=I-g_{ij}^{*}(g_{ij}^{*})^T.
$$

用 $$P_{g_{ij}^{*}}$$ 作为图边的矩阵权重构造方位拉普拉斯矩阵，并按 leader 和 follower 划分为

$$
\mathcal B=
\begin{bmatrix}
\mathcal B_{\ell\ell}&\mathcal B_{\ell f}\\
\mathcal B_{f\ell}&\mathcal B_{ff}
\end{bmatrix}.
$$

当 $$\mathcal B_{ff}$$ 非奇异时，follower 目标位置和速度为

$$
p_f^{*}=-\mathcal B_{ff}^{-1}\mathcal B_{f\ell}p_\ell,
\qquad
v_f^{*}=-\mathcal B_{ff}^{-1}\mathcal B_{f\ell}v_\ell.
$$

代码不显式计算矩阵逆，而是使用 `numpy.linalg.solve` 求解相应线性方程。

### 控制律

#### 恒定 leader 速度

当 leader 速度为常值时，follower 使用论文式 (7)：

$$
u_i=-\sum_{j\in\mathcal N_i}P_{g_{ij}^{*}}
\left[k_p(p_i-p_j)+k_v(v_i-v_j)\right].
$$

在矩阵形式下，follower 加速度可写为

$$
\dot v_f=-k_p(\mathcal B_{ff}p_f+\mathcal B_{f\ell}p_\ell)
-k_v(\mathcal B_{ff}v_f+\mathcal B_{f\ell}v_\ell).
$$

#### 时变 leader 速度

当 leader 速度时变时，复现使用论文式 (10)：

$$
u_i=-K_i^{-1}\sum_{j\in\mathcal N_i}P_{g_{ij}^{*}}
\left[k_p(p_i-p_j)+k_v(v_i-v_j)-\dot v_j\right],
\qquad
K_i=\sum_{j\in\mathcal N_i}P_{g_{ij}^{*}}.
$$

由于 follower 之间存在加速度耦合，代码同步求解

$$
\mathcal B_{ff}\dot v_f+\mathcal B_{f\ell}\dot v_\ell
=-k_p(\mathcal B_{ff}p_f+\mathcal B_{f\ell}p_\ell)
-k_v(\mathcal B_{ff}v_f+\mathcal B_{f\ell}v_\ell),
$$

而不是将 follower 邻居加速度设为零。该实现与论文式 (10) 的矩阵推导一致。

### 编队、拓扑与初值

主要动画实验使用二维正八边形目标编队，共有 8 个 agent，包含 2 个 leader 和 6 个 follower。信息图由八边形周边和两组 leader 扇形对角边构成，共包含 18 条无向边。该构型下

$$
\lambda_{\min}(\mathcal B_{ff})=0.0761205>0,
\qquad
\operatorname{cond}(\mathcal B_{ff})=50.1846,
$$

因此目标 follower 位置唯一。

统一实验参数为：

| 参数 | 数值 |
|---|---:|
| agent 数 | 8（2 leader + 6 follower） |
| $$k_p$$ | 1.0 |
| $$k_v$$ | 4.0 |
| follower 初始偏差幅度 | 4.0 |
| follower 初速度 | 0.0 |
| 随机种子 | 7 |
| 仿真时长 | 60 s |
| RK4 步长 | 0.01 s |

### leader 轨迹

静止案例中，leader 位置保持不变。恒速案例中，两个 leader 使用不随时间变化的速度，使编队同时匀速平移和线性改变尺度。

时变案例使用以下质心和尺度轨迹：

$$
c(t)=\begin{bmatrix}0.18t\\2.2\sin(0.18t)\end{bmatrix},
\qquad
s(t)=1+0.20\sin(0.12t).
$$

该设计使 leader 的速度和加速度随时间变化，同时保证 $$s(t)>0$$ 且所有目标方位不变。

### 评价指标

复现计算以下指标：

- 总方位误差：$$e_g(t)=\sum_{(i,j)\in\mathcal E}\|g_{ij}(t)-g_{ij}^{*}\|$$。
- follower 位置跟踪误差：$$e_p(t)=\|p_f(t)-p_f^{*}(t)\|_2$$。
- follower 速度跟踪误差：$$e_v(t)=\|v_f(t)-v_f^{*}(t)\|_2$$。
- 方位误差进入并保持在 $$10^{-3}$$ 以下的时间。
- 仿真过程中任意两 agent 的最小距离。

### 复现结果

| 案例 | 控制律 | leader 速度范围 | 最终 $$e_g$$ | 最终 $$e_p$$ | 最终 $$e_v$$ | $$e_g<10^{-3}$$ 并保持的时间 |
|---|---|---:|---:|---:|---:|---:|
| leader 静止 | 式 (7) | `0` | `2.48e-6` | `1.40e-5` | `4.61e-6` | `33.95 s` |
| leader 恒速 | 式 (7) | `0.1841–0.1902` | `1.17e-5` | `1.21e-4` | `1.24e-5` | `34.51 s` |
| leader 速度时变 | 式 (10) | `0.1800–0.4772` | `5.87e-7` | `8.93e-7` | `2.39e-7` | `33.40 s` |

三类案例的总方位误差、位置跟踪误差和速度跟踪误差均收敛到接近零。恒速 leader 案例验证了式 (7) 对平移和线性尺度变化的跟踪能力；时变 leader 案例中最大 leader 加速度约为 `0.0761`，式 (10) 仍使最终位置误差降至 `8.93e-7`，说明加速度前馈有效消除了时变 leader 运动对误差动力学的影响。

三类实验中观测到的最小两体距离均为 `1.0653`。该数值仅表示本次初值和参数下没有碰撞，不是一般性防碰保证。

### 可视化结果

动画左图为编队轨迹，中图为所有 agent 的速度模，右图为总方位误差。leader 统一用红色表示，follower 统一用蓝绿色表示，不区分具体编号。

#### 静止 leader

![静止 leader 编队形成](results/formation_to_target.gif)

#### 恒定 leader 速度

![恒定 leader 速度](results/constant_velocity_demo.gif)

#### 时变 leader 速度

![时变 leader 速度](results/time_varying_velocity_demo.gif)

### 数值验证

- 将 RK4 步长从 `0.01 s` 减半到 `0.005 s` 后，恒速案例的最终位置误差从 `1.2137473884e-4` 变为 `1.2137473883e-4`，时变案例从 `8.9283861e-7` 变为 `8.9283957e-7`。
- 恒速和时变目标轨迹中，目标方位相对初始值的最大漂移分别为 `1.14e-15` 和 `1.57e-15`，仅为浮点舍入误差。
- 所有 GIF 均为 `1387×456`、175 帧并循环播放；抽查首帧、中间帧和末帧后未发现空白帧或坐标、图例重叠。

### 与原论文实验的关系

`bearing_formation.py` 中的 4-agent 案例参考原论文图 4 的二维方阵、2 leader + 2 follower 以及 $$k_p=0.5,k_v=2$$。由于论文没有列出完整初值、leader 速度和全部拓扑数值，该实验是对核心控制律与收敛性的独立复现，不是原图的像素级重建。

8-agent 正八边形动画是在同一方位控制理论上扩展的可视化实验。它用于展示更多 agent 时的编队形成、恒速机动和时变机动，并非对原论文三维立方体轨迹的逐点复制。

## 局限性

- 仿真使用理想双积分器，没有加入传感器噪声、通信延迟、丢包或非完整机器人动力学。
- 式 (10) 在代码中通过 $$\mathcal B_{ff}$$ 线性系统同步求解 follower 加速度，数学上与论文推导一致，但不是实际通信网络上的异步分布式软件实现。
- 当前复现未实现论文的积分抗常值扰动、饱和控制和防碰充分条件。
- 时变 leader 轨迹是为复现而设计的可行平移-尺度轨迹，不是原文未公开参数的猜测。

## 参考文献

Shiyu Zhao and Daniel Zelazo, “Translational and Scaling Formation Maneuver Control via a Bearing-Based Approach,” arXiv:1506.05636v2, 2015.
