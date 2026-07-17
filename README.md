# 基于方位的多智能体编队控制复现

本仓库复现 Zhao 和 Zelazo 论文 *Translational and Scaling Formation Maneuver Control via a Bearing-Based Approach* 中的核心算法。代码使用 Python 实现基于方位拉普拉斯矩阵的 leader-follower 编队控制，并给出静止、恒定速度和时变速度三类 leader 运动的仿真、误差指标和 GIF 动画。

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

最小两体距离均为 `1.0653`。该数值仅表示本次初值和参数下没有碰撞，不是一般性防碰保证。

## 可视化结果

动画左图为编队轨迹，中图为所有 agent 的速度模，右图为总方位误差。leader 统一用红色表示，follower 统一用蓝绿色表示，不区分具体编号。

### 静止 leader

![静止 leader 编队形成](results/formation_to_target.gif)

### 恒定 leader 速度

![恒定 leader 速度](results/constant_velocity_demo.gif)

### 时变 leader 速度

![时变 leader 速度](results/time_varying_velocity_demo.gif)


## 参考文献

Shiyu Zhao and Daniel Zelazo, “Translational and Scaling Formation Maneuver Control via a Bearing-Based Approach,” arXiv:1506.05636v2, 2015.
