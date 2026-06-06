# 🟡 M1-Riemannian FM 与 SE(3) 生成

> **这是 FM 第二次出现。** 在 C3 里你掌握了 R^d 里的 FM。现在把它搬到 SE(3) 这个弯曲流形上，让它生成有效的机械臂位姿和运动轨迹。
>
> **预计时间：** 3 周
>
> **状态：** 📝 待学习

---

## 1. 从 R^d 到流形——改了什么？

| | 欧氏 FM (C3) | Riemannian FM (M1) |
|---|---|---|
| **空间** | R^d | 黎曼流形 M（如 SE(3)） |
| **条件路径** | x_t = (1-t)x₀ + tx₁ | x_t = exp_{x₀}(t · log_{x₀}(x₁)) |
| **条件向量场** | v = x₁ - x₀ | u_t = log_{x_t}(x₁) / (1-t) |
| **向量场所在空间** | R^d | T_{x_t} M（切空间） |
| **损失距离** | L2 | 黎曼度量 g_{x_t}(v_pred, v_true) |

核心 insight：**在流形上，你沿着测地线走而不是直线。向量场活在切空间里，不是 R^d 里。**

---

## 2. SE(3) 上的 FM 完整实现

```python
import torch, torch.nn as nn
from pytorch3d.transforms import se3_log_map, se3_exp_map

class SE3FlowMatching(nn.Module):
    def __init__(self, d_hidden=256, sigma_r=1.0, sigma_t=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, d_hidden), nn.ReLU(),   # 6(se3) + 1(t)
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, 6))               # output in se(3)
        self.sigma_r = sigma_r   # 旋转单位：弧度
        self.sigma_t = sigma_t   # 平移单位：米

    def forward(self, T, t):
        xi = se3_log_map(T)                        # localize at identity
        return self.net(torch.cat([xi, t], dim=-1))

    def weighted_mse(self, v_pred, v_true):
        """SE(3) 的加权度量——统一弧度和米的量纲"""
        loss_r = ((v_pred[:,:3]-v_true[:,:3])**2).sum(-1) / self.sigma_r**2
        loss_t = ((v_pred[:,3:]-v_true[:,3:])**2).sum(-1) / self.sigma_t**2
        return (loss_r + loss_t).mean()

    def loss(self, T0, T1):
        B = T0.size(0)
        t = torch.rand(B, 1, 1)
        delta = se3_log_map(T0.inverse() @ T1)      # log(T0^{-1}·T1)
        Tt = T0 @ se3_exp_map(t * delta)             # 测地线插值
        v_target = delta.squeeze(1) / (1-t.squeeze(-1)+1e-8)
        return self.weighted_mse(self(Tt, t.squeeze(-1)), v_target)

    @torch.no_grad()
    def sample(self, T0, n_steps=50):
        T = T0.clone()
        dt = 1.0/n_steps
        for i in range(n_steps):
            t = torch.full((T.size(0),1), i*dt)
            v = self(T, t)                          # se(3) vector
            T = T @ se3_exp_map(dt * v.unsqueeze(1))  # Euler on SE(3)
        return T
```

---

## 3. Reflow —— 让路径变直

### 问题

测地线路径也可能交叉。两个成功的抓取位姿之间，中间 t≈0.5 处两条测地线可能撞在一起 → 需要学复杂的向量场来"绕开"。

### 解决

```
Reflow 算法：
  1. 训练初始 FM 模型 v^(0)
  2. 用 v^(0) 生成配对 (T₀, T̂₁)（从 p₀ 采样，跑 ODE 到 t=1）
  3. 用这些配对重训 v^(1)（路径已更直）
  4. 重复 2-3

极限：路径完全笔直 → 1 步 ODE 即可生成

Reflow 在 SE(3) 上同样适用——
  因为操作的都是 (T₀, T̂₁) 配对，与空间的性质无关
```

---

## 4. 蒸馏 —— 多步教师 → 单步学生

```
你的小脑回路需要 100+ Hz → FM 采样 50 步不够快

蒸馏方案（参考 Flow to One Step, 2026.03）：
  教师：CFM，50 步 ODE → 高质量样本
  学生：单步预测，x_1 = f_θ(x_0, observation)
  
  训练：学生输出 ∥f_θ(x₀, obs) - 教师采样结果∥²
        + IMLE (Implicit Maximum Likelihood Estimation) 保留多模态性

结果：1 步推理 → 1000Hz 可行
代价：单步的生成质量和多样性略逊于多步
```

---

## 5. FM 作为机器人策略

### 策略 = 条件 FM

```
标准 FM：     x₀ ~ p₀(noise)  →  ODE →  x₁ ~ p_data
FM 策略：     x₀ = 当前位姿 + 噪声  →  ODE →  x₁ = 未来位姿轨迹
              条件 = 当前观察（视觉、力觉、触觉）

观察 o 作为条件注入网络：
  v_θ(T_t, t, o)  而不是 v_θ(T_t, t)

条件如何注入：
  → Cross-attention: 观察 embedding 作为 K/V, 位姿 token 作为 Q
  → FiLM: 观察 → scale + shift 参数
  → Concatenation: [位姿, 观察, t] → MLP
```

### 动作空间设计

```
乘积流形：M = SE(3) × R^6 × R^{fingers}
            位姿     力/力矩   手指关节

每个子流形的测地线：
  → SE(3): 李群测地线
  → R^6: 直线（力不需要弯曲路径）
  → R^{fingers}: 直线（或考虑关节限位约束）

联合 FM：
  条件路径在乘积流形上 = 各分量的测地线之积
  向量场也取乘积 = 各分量独立的条件向量场
  但！耦合发生在网络内部（通过 attention/concatenation）
```

---

## 📚 学习资源

### 视频

| 资源 | 说明 |
|------|------|
| 🎥 Yaron Lipman ICLR 2023 + 后续 guest lectures | FM 基础（C3 看过的，回顾） |
| 🎥 Frank Dellaert / Cyrill Stachniss | SE(3) 几何（C2 看过的） |

### 必读论文（按问题组织）

**问题：FM 怎么搬到弯曲空间？**

| # | 论文 | 为什么读 |
|---|------|----------|
| ⭐⭐⭐ | Chen & Lipman, "Flow Matching on General Geometries", ICLR 2024 | Riemannian FM 的定义和公式 |
| ⭐⭐ | Kapusniak et al., "Metric Flow Matching", 2024 | 自动学测地线权重 |

**问题：FM 怎么生成机器人动作？**

| # | 论文 | 为什么读 |
|---|------|----------|
| ⭐⭐⭐ | Braun et al., "Riemannian Flow Matching Policy", CoRL 2024 | FM 策略的直接参考 |
| ⭐⭐⭐ | Black et al., "π₀: A VLA Flow Model", RSS 2025 | 大规模 FM 策略的标杆 |
| ⭐⭐⭐ | Chi et al., "Diffusion Policy", RSS 2023 | 你的 baseline（把 Diffusion 换成 FM） |
| ⭐⭐ | Urain et al., "SE(3)-DiffusionFields", CoRL 2023 | SE(3) Diffusion 做抓取 |

**问题：怎么让 FM 更快？**

| # | 论文 | 为什么读 |
|---|------|----------|
| ⭐⭐⭐ | "From Flow to One Step", 2026.03 | 蒸馏到单步——小脑回路使能技术 |
| ⭐⭐⭐ | Streaming Flow Policy, CoRL 2025 | 流式动作输出 |
| ⭐⭐ | FlowRAM, CVPR 2025 | <4 步推理 |

---

## ✍️ 写作练习

### L1 费曼笔记

- [ ] 用大白话解释：**Riemannian FM 和普通 FM 的损失函数差在哪？为什么不能直接在 SE(3) 上用 L2 距离？**

### L2 概念串联

- [ ] 画文字图：C2 的 SE(3) 测地线 → C3 的 FM 条件路径 → M1 的 Riemannian FM 损失 → Reflow → 蒸馏。标注每个概念的"输入"和"输出"。

### L3 设计性写作

→ 写到 [I4-累积设计文档](/robotics-learning-map/I4-累积设计文档) Section 2：你的 SE(3) + 力联合生成方案

---

## 🚫 常见弯路

- ❌ 自己去实现 SE(3) 的 exp/log。用 pytorch3d 或 geomstats 就好。
- ❌ 在没有理解 CFM 之前就看 Riemannian FM。必须按 C3 → M1 顺序。

---

## 🔗 相关笔记

- [C2-SE3几何入门](/robotics-learning-map/C2-SE3几何入门) — SE(3) 的几何基础
- [C3-Flow-Matching基础](/robotics-learning-map/C3-Flow-Matching基础) — FM #1（欧氏空间）
- [I1-系统架构与部署](/robotics-learning-map/I1-系统架构与部署) — FM #3（蒸馏部署）
- [I4-累积设计文档](/robotics-learning-map/I4-累积设计文档) — L3 输出写到这里
- [写作工具箱](/robotics-learning-map/写作工具箱)
