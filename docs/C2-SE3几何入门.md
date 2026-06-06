# 🔴 C2-SE3 几何入门

> **核心问题：** 为什么机械臂位姿不在 R^6 里？如果不理解这点，你的 FM 会生成无效的旋转矩阵（行列式 ≠ 1 的"旋转"）。
>
> **预计时间：** 1 周
>
> **状态：** 📝 待学习

---

## 1. 问题：R^6 里的"位姿"其实是错的

```
常见错误做法：
  pose = (x, y, z, roll, pitch, yaw) ∈ R^6
  在 R^6 里做插值、做概率分布、做 FM

问题：
  1. 两个旋转之间 L2 距离 ≠ 旋转角度距离
     (roll=179°, pitch=0°,  yaw=0°) 和 (roll=-179°, pitch=0°, yaw=0°)
     → 几乎一样的姿态，L2 距离却巨大

  2. 线性插值两个旋转矩阵 → 结果不在 SO(3)（行列式 ≠ 1）
     R_interp = 0.5*R₁ + 0.5*R₂ → R_interp 不是正交矩阵

  3. 欧拉角有万向节锁——不是所有姿态都能唯一表示
```

---

## 2. SO(3): 旋转空间

### 几何直觉

```
SO(3) 可以想象成 4 维球面 S³ 的一半。
    → 3 维的旋转空间嵌入在 4 维空间里
    → 它是一个弯曲的、紧致的 3 维流形

单位四元数表示 SO(3) 最自然：
  q = [w, x, y, z],  w²+x²+y²+z²=1
  q 和 -q 表示同一个旋转（对径点等价 → S³ 的一半）
```

### 核心运算

```
so(3): 3×3 反对称矩阵（"旋转速度"的空间，3维向量空间）

指数映射 exp: so(3) → SO(3)
  exp(ω̂) = I + sin(θ)/θ · ω̂ + (1-cos(θ))/θ² · ω̂²
  → 物理意义：绕轴 ω 转 θ 弧度

对数映射 log: SO(3) → so(3)
  log(R) = θ/(2sinθ) · (R - Rᵀ)
  → 物理意义：从旋转矩阵中提取"旋转了 θ 弧度，绕轴 ω"

测地线距离（两个旋转之间的"最短角距离"）：
  d(R₁, R₂) = || log(R₁ᵀ · R₂) ||
```

---

## 3. SE(3): 刚体运动空间

### 表示

```
SE(3) 是 6 维流形：3 维旋转 × 3 维平移（但它们是耦合的——半直积）

齐次矩阵表示：
  T = [R  t]   R ∈ SO(3), t ∈ R³
      [0  1]

se(3): 6 维向量空间（"运动旋量"的空间）
  ξ = [ω, v]   ω ∈ so(3)（旋转速度），v ∈ R³（平移速度）
```

### 核心运算

```
exp: se(3) → SE(3)
  把"小运动旋量"变成刚体变换矩阵

log: SE(3) → se(3)
  把刚体变换矩阵变成运动旋量

测地线插值（SE(3) 上的"直线运动"）：
  T(τ) = T₀ · exp(τ · log(T₀⁻¹ · T₁))

  → 这保证插值结果始终在 SE(3) 上
  → 旋转走最短弧，平移走直线
  → 这是你 FM 条件路径的基础

测地线距离（需要加权——弧度和米量纲不同）：
  d(T₁, T₂) = || log(T₁⁻¹ · T₂) ||_W
  ||(ω, v)||²_W = ||ω||²/σ_r² + ||v||²/σ_t²
```

---

## 4. 对你 FM 设计的影响

```
在 R^d 里做 FM：
  x_t = (1-t)·x₀ + t·x₁         ← 线性插值
  v_target = x₁ - x₀             ← 恒定速度

在 SE(3) 上做 FM：
  T_t = T₀ · exp(t · log(T₀⁻¹·T₁))   ← 测地线插值
  v_target = log(T_t⁻¹·T₁)/(1-t)      ← 在 se(3) 中的条件向量场

你的网络输出：不是 R^6 向量，而是 se(3) 中的切向量。
损失函数中的距离：不是 L2，是 SE(3) 的加权测地线度量。
```

---

## 📚 学习资源

| 资源 | 说明 |
|------|------|
| 📄 [Joan Solà: "A micro Lie theory for state estimation in robotics"](https://arxiv.org/abs/1812.01537) | **机器人 SE(3) 必读。** 从零推导 exp/log/Jacobian，有代码示例 |
| 🎥 [Frank Dellaert: Lie Groups for Robotics (GTSAM)](https://www.youtube.com/results?search_query=gtsam+lie+groups+robotics) | 直观的视觉讲解 |
| 🎥 [Cyrill Stachniss: 3D Transformations in Robotics](https://www.youtube.com/results?search_query=cyrill+stachniss+3d+transformations) | 德国波恩大学，讲得特别清楚 |
| 📄 Ethan Eade: "Lie Groups for 2D and 3D Transformations" | 比 Solà 更短的入门 |
| 📦 `pytorch3d.transforms` (so3_log_map, se3_log_map) | Python 库，直接调 |

### 先读这篇

⭐⭐⭐ Solà, "A micro Lie theory", Sections 1-4 and 7-9 (跳过深奥的李群数学证明部分，重点看 SO(3) 和 SE(3) 的 exp/log 公式和直觉)

---

## ✍️ 写作练习

### L1 费曼笔记

- [ ] 用大白话解释：**"在 R^6 里插值两个位姿"和"在 SE(3) 上插值两个位姿"有什么不同？** 假设你在用 FM 给机械臂生成抓取轨迹，走错了会怎样？

### L2 概念串联

- [ ] 画文字图：C1 的 FK → SE(3) 位姿 → log → se(3) twist → J → 关节速度。把 C1 和 C2 串起来。

---

## 🚫 常见弯路

- ❌ 去学完整的李群表示论（roots, weights, Dynkin diagrams）。对机器人来说 SO(3) 和 SE(3) 够了。
- ❌ 纠结四元数和旋转矩阵哪个好。用旋转矩阵推导，用四元数存储——两者在 3×3 矩阵和 4 维向量间可无损互转。

---

## 🔗 相关笔记

- [C1-经典机器人学](/C1-经典机器人学) — 运动学的 SE(3) 基础
- [M1-Riemannian-FM与SE3生成](/M1-Riemannian-FM与SE3生成) — 在 SE(3) 上做 FM（进阶）
- [写作工具箱](/写作工具箱)
