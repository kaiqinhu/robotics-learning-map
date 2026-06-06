# 🔴 C3-Flow Matching 基础

> **核心问题：** 给定一个简单分布 p₀ 和一个目标分布 p₁，学一个向量场把 p₀ "推"成 p₁。为什么 FM 比 Diffusion 更适合机器人控制？
>
> **这是 FM 在 R^d 里的完整机制。SE(3) 上的版本在 M1。**
>
> **预计时间：** 2 周
>
> **状态：** 📝 待学习

---

## 1. 一句话直觉

```
FM = 给空间中每个点指定一个"水流速度"，让噪声顺着水流漂到数据。

  Diffusion: "每步加一点噪声，学怎么去噪"（随机漫步，1000步）
  FM:       "学一个风场，直接吹过去"（确定性的流，10步）
```

---

## 2. Continuous Normalizing Flow (CNF) — FM 的前身

```
CNF 用 ODE 变换概率分布：

  dx/dt = v_t(x),   x(0) ~ p₀

如果 v_t(x) 选对了，x(1) ~ p₁。

"选对"的条件：p_t(x) 满足连续性方程（Continuity Equation）
  ∂p_t/∂t + ∇·(p_t · v_t) = 0     ← 概率质量守恒

问题：学 v_t 需要知道 p_t —— 但我们不知道（密度估计本身就很难）
→ CNF 训练很难！需要反复解 ODE + 算最大似然
```

---

## 3. Conditional Flow Matching (CFM) — 关键突破

### 核心 trick

```
没法算边际向量场 u_t(x)（因为我们不知道 p_t(x)）
→ 换一个思路：学条件向量场 u_t(x | x₁)

CFM 损失：
  L_CFM = E_{t, x₀~p₀, x₁~p₁} || v_θ(x_t, t) - u_t(x_t | x₁) ||²

条件路径（最简单的——线性插值）：
  x_t = (1-t)·x₀ + t·x₁
  u_t(x_t | x₁) = x₁ - x₀   （恒定速度——最简单）

关键定理：∇L_CFM = ∇L_FM  ← 两个梯度完全一样！
→ 优化 CFM loss 等价于优化原始 FM loss
→ "免费"解决了 CNF 的训练难题
```

### 完整训练流程

```
1. 采样: x₀ ~ p₀ (高斯), x₁ ~ p₁ (数据)
2. 采样时间: t ~ Uniform(0,1)
3. 构造条件样本: x_t = (1-t)·x₀ + t·x₁
4. 条件速度标签: u_target = x₁ - x₀
5. 网络预测: v_pred = v_θ(x_t, t)
6. 损失: MSE(v_pred, u_target)  ← 简单的回归！
```

### 完整采样流程

```
1. 采样初始点: x(0) ~ p₀
2. 解 ODE: dx/dt = v_θ(x, t), t: 0→1
   - 可以用 Euler: x_{k+1} = x_k + Δt · v_θ(x_k, t_k)
   - 或用更好的 solver (RK4, Dopri5)
3. x(1) 就是生成的样本
```

---

## 4. FM vs Diffusion：实质区别

| | Diffusion (DDPM) | Flow Matching |
|---|---|---|
| **训练目标** | 预测噪声 ε | 预测速度 v = x₁-x₀ |
| **采样方式** | Markov 链去噪 | 解 ODE |
| **路径性质** | 弯曲的、随机 | 可设计为直线 |
| **步数** | 通常 100-1000 | 10-50 (直线路径) |
| **可逆性** | 需单独学反向过程 | 天然可逆 |
| **CFG** | ε_cfg = ε_∅ + w(ε_c - ε_∅) | v_cfg = v_∅ + w(v_c - v_∅) |
| **数学统一性** | FM 是更一般的框架，Diffusion 是 Gaussian 路径的特殊情况 |

---

## 5. 几个重要的设计选择

### 条件路径的选择

```
线性插值：        x_t = (1-t)·x₀ + t·x₁
  → 最简单，但对某些分布路径会交叉

VP (Variance Preserving)：  x_t = α_t·x₀ + β_t·x₁
  → Diffusion 等价形式

Optimal Transport (OT)：    x_t 通过 OT 配对决定
  → 路径交叉最少，但需要额外计算 OT
```

### ODE Solver 的选择

```
Euler（1 阶）：    简单，但不精确，需要更多步数
RK4（4 阶）：      精确很多，每步 4 次 forward
Dopri5（自适应）： 自动调整步长，弯曲处用小步
```

---

## 6. 动手：从零实现 2D FM

```python
import torch, torch.nn as nn
from torchdiffeq import odeint

class FlowMatching(nn.Module):
    def __init__(self, dim=2, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim+1, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dim))

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=-1))

    def loss(self, x0, x1):
        t = torch.rand(x0.size(0), 1)
        xt = (1-t)*x0 + t*x1
        v_target = x1 - x0
        return ((self(xt, t) - v_target)**2).mean()

    @torch.no_grad()
    def sample(self, x0, steps=50):
        dt = 1.0/steps
        x = x0
        for i in range(steps):
            t = torch.full((x.size(0),1), i*dt)
            x = x + dt * self(x, t)
        return x
```

---

## 📚 学习资源

| 资源 | 说明 |
|------|------|
| 🎥 [ICLR 2025: "Flow With What You Know"](https://iclr-blogposts.github.io/2025/blog/flow-with-what-you-know/) | **首选。** 物理直觉——把 FM 比作风场，有可运行 notebook |
| 🎥 [ICLR 2025: "A Visual Dive into Conditional Flow Matching"](https://iclr-blogposts.github.io/2025/blog/conditional-flow-matching/) | 数学最完整的教程，每公式配可视化 |
| 📄 [BLOPIG: "Getting In the Flow"](https://www.blopig.com/blog/2025/09/getting-in-the-flow-how-to-flow-match/) | 最友好的 walkthrough |
| 📄 [TuringPost: "What is Flow Matching?"](https://www.turingpost.com/p/flowmatching) | 零基础可读 |
| 💻 [Vicki Mu (MIT): "FM from Scratch"](https://ai.gopubby.com/from-noise-to-structure-building-a-flow-matching-model-from-scratch-f1ca12b31602) | **代码首选。** 完整 PyTorch + Colab |
| 🎥 [Yaron Lipman: ICLR 2023 Oral](https://www.youtube.com/results?search_query=lipman+flow+matching+iclr+2023) | 15 分钟，FM 核心思想 |

### 先读 3 篇

1. ⭐⭐⭐ Lipman et al., "Flow Matching for Generative Modeling", ICLR 2023
2. ⭐⭐⭐ Liu et al., "Rectified Flow", ICLR 2023
3. ⭐⭐⭐ "Flow With What You Know" (ICLR 2025 blog)

---

## ✍️ 写作练习

### L1 费曼笔记

- [ ] 用大白话解释：**CFM 为什么能用条件路径的向量场替代真实向量场？这个"等价性"是怎么来的？**

### L2 概念串联

- [ ] 画文字图：Diffusion (DDPM) → Score-based SDE → PF-ODE → CNF → CFM → Rectified Flow。标注每一步解决了上一个的什么痛点。

---

## 🚫 常见弯路

- ❌ 去推 FM 等价于 Diffusion 的完整数学证明。知道结论、能说清楚区别就够了。
- ❌ 纠结 ODE solver 的数值分析细节。会用 Euler 和 RK4，能说出 Dopri5 的原理即可。

---

## 🔗 相关笔记

- [M1-Riemannian-FM与SE3生成](/M1-Riemannian-FM与SE3生成) — FM 从 R^d 搬到 SE(3)（FM #2）
- [I1-系统架构与部署](/I1-系统架构与部署) — FM 蒸馏到实时（FM #3）
- [写作工具箱](/写作工具箱)
