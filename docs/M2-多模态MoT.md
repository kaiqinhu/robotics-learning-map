# 🟡 M2-多模态 MoT

> **核心问题：** 视觉(30Hz, 高维)、力觉(1000Hz, 6D)、触觉(60Hz, 图像)、位姿(100Hz, 6D)、关节(1000Hz)——频率不同、维度差距大、有时缺失。怎么用一个 MoE Transformer 处理所有这些模态？
>
> **预计时间：** 2 周
>
> **状态：** 📝 待学习

---

## 1. 为什么需要 MoE？

```
单 FFN 的问题：
  → 所有 token 共享同一个 FFN
  → 视觉 token 和力觉 token 互相干扰
  → "接触时应该更相信触觉"——单 FFN 做不到这种条件路由

MoE 的优势：
  → 不同模态走不同 expert
  → 接触状态变化 → Router 自动切换专家权重
  → 增加总参数但不增加计算量（稀疏激活）
```

---

## 2. MoE 核心机制

```
输入 token x ⟶ Router (线性层 + softmax) ⟶ Top-k expert

Router:  r = softmax(W_r · x)          ← r ∈ R^{n_experts}
选择:    top-k indices of r
输出:    y = Σ_{i∈topk} r_i · Expert_i(x)

Load Balancing Loss（必须——否则所有 token 可能都用同一个 expert）：
  L_balance = n_experts² · Σ f_i · p_i
  f_i = 实际分配到 expert i 的 token 比例
  p_i = router 分配给 i 的平均概率
```

### 路由策略对比

| 策略 | 做法 | 你的场景 |
|------|------|----------|
| Token Choice | 每个 token 选 top-k | 基础方案 |
| 模态路由 | 按 token 来源直接分配 | **核心。** 视觉→Vision Expert, 力→Force Expert |
| 混合路由 | 模态路由(第一层) + 内容路由(第二层) | **你的设计空间。** 既保证模态隔离，又允许跨模态共享 |

---

## 3. 多模态设计空间

### 频率对齐

```
视觉 30Hz   ─→ 重复/插值 ─→
力觉 1000Hz ─→ 降采样     ─→  统一 100Hz → Transformer
触觉 60Hz   ─→ 插值       ─→
位姿 100Hz  ─→ 保持       ─→
```

### 跨模态注意力

```
模态内 Self-Attn：  触觉 token 互相看 → 时序一致性
跨模态 Cross-Attn：  力 token 看触觉 token → "我感觉到接触了吗？"
                    触觉 token 看视觉 token → "我摸到的是视觉看到的那个东西吗？"
```

### 专家设计

```
独立专家：  Vision Expert / Force Expert / Tactile Expert
共享专家：  Shared Expert（处理跨模态模式）
总专家数建议： 6-12（太多 → load balance 难，太少 → 不够专精）
```

### 缺失模态处理

```
触觉传感器坏了 → 触觉 token 用 learnable [MISSING] embedding 替代
训练技巧：随机 drop 真机数据中的力/触觉 → 模型学会优雅退化
```

---

## 4. 动作 Token 化

| 方法 | 做法 | 适用 |
|------|------|------|
| Discretization | 连续动作 → bins (256/dim) | RT-2 式 |
| Action Chunk | MLP 压缩动作序列 → 一个 token | Diffusion Policy |
| FM Head | Transformer 输出向量场参数 | **你的方案。** π₀ 式 |

---

## 📚 学习资源

| 资源 | 说明 |
|------|------|
| 🎥 [Yannic Kilcher: Switch Transformer](https://www.youtube.com/watch?v=iAR8LkkMMIM) | MoE 论文精读，讲透路由和 load balance |
| 🎥 [Andrej Karpathy: "Let's build GPT"](https://www.youtube.com/watch?v=kCc8FmEb1nY) | Transformer 基础（C3 已看，回顾） |
| 📄 [HuggingFace: MoE 博客](https://huggingface.co/blog/moe) | 图文并茂，从 Switch 到 Mixtral |

### 先读 3 篇

1. ⭐⭐⭐ Fedus et al., "Switch Transformers", JMLR 2022
2. ⭐⭐ Riquelme et al., "Scaling Vision with Sparse MoE", NeurIPS 2021（视觉 MoE——对触觉专家设计有参考）
3. ⭐⭐ Jiang et al., "Mixtral of Experts", 2024（工业 MoE 实践）

---

## ✍️ 写作练习

### L1 费曼笔记

- [ ] 用大白话解释：**Load balancing loss 是干什么的？如果不用它会有什么后果？**

### L2 概念串联

- [ ] 画文字图：5 种模态 → 各自的 encoder → 统一的 token 序列 → Router → 哪些专家应该共享、哪些独立 → 跨模态 attention 发生在哪里？

### L3 设计性写作

→ 写到 [I4-累积设计文档](/I4-累积设计文档) Section 3：你的多模态融合架构

---

## 🔗 相关笔记

- [M1-Riemannian-FM与SE3生成](/M1-Riemannian-FM与SE3生成) — FM Head 接在 Transformer 后面
- [M3-触觉表征与力预测](/M3-触觉表征与力预测) — 触觉/力 encoder 设计
- [I4-累积设计文档](/I4-累积设计文档)
- [写作工具箱](/写作工具箱)
