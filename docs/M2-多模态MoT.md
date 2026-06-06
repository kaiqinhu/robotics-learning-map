# 🟡 M2-多模态 MoT (Mixture of Transformers)

> **预计时间：** 2 周
>
> **状态：** 📝 待学习

---

## 背景：多模态不是"拼积木"

机器人策略的输入不是一张图+一句话——它是五路完全不同的信号：
```
视觉    30Hz   高维像素      → 场景里有什么
力觉   1000Hz   6D wrench    → 接触力多大、什么方向
触觉    60Hz   触觉图像      → 滑不滑、什么材质
位姿   100Hz   6D SE(3)      → 手在哪、目标在哪
语言   离散    token序列     → 任务是什么
```

把这五路信号塞进一个 Dense Transformer？视觉 token 有几百个，力觉 token 只有一两个——**高维视觉会把低维力觉完全淹没。** 这就是为什么传统 VLM 做 robot policy "看图厉害，动作不准"。

**MoT (Mixture of Transformers)** 是 2024-2026 年间出现的关键解法：不给所有模态共享一套参数，而是每个模态拥有独立的 QKV、FFN、LayerNorm。它不是 MoE（在 FFN 里选 expert），而是在 **Block 级** 做模态分家。

Meta 提出了框架，NVIDIA Cosmos 3 用它做了全模态世界模型，腾讯 HY-Embodied 用它做了具身 VLA。你的模型也需要一个 MoT backbone。

## 学完这一页你能

- 用一句话区分 MoT 和 MoE（Block 级 vs FFN 级；确定性模态 mask vs 学习的 Router）
- 画出 MoT 的架构图：每个模态的 QKV+FFN+LN 各自独立，全局 attention 共享
- 解释为什么 MoT 天然适合具身多模态——频率对齐、模态隔离、力觉不被视觉稀释
- 说出 Cosmos 3 双塔的单向注意力设计及其原因
- 设计你自己的 MoT 方案：哪些模态独立 Transformer？cross-attention 在哪几层？

## 核心驱动问题

1. **MoT 和 MoE 到底差在哪？** 为什么说 MoE 是 FFN 级、MoT 是 Block 级？确定性模态 mask 和学习 Router 的本质区别是什么？
2. **为什么视觉 token 会"挤占"力觉 token 的参数空间？** 频率不同、维度不同、信息密度不同——这三个"不同"各自造成什么后果？
3. **Cosmos 3 为什么设计成单向注意力（AR→DM）？** AR token 不被 DM token 更新——这个设计在机器人策略里需要吗？
4. **MoT 的消融实验说 FFN 解耦贡献最大、QKV 第二、LN 最小——对你的设计有什么启示？** 如果算力紧，先分什么？
5. **在你的场景里，力觉 token 只有 1-2 个——给它们独立 QKV 性价比高吗？** 还是共享 QKV + 独立 FFN 就够了？

---

> **核心问题：** 视觉(30Hz, 高维)、力觉(1000Hz, 6D)、触觉(60Hz, 图像)、位姿(100Hz, 6D)、语言(离散)——不同模态信息密度差异悬殊。一个 Dense Transformer 让所有 token 共享参数 → 视觉挤占语言、力觉被稀释。**MoT 给每种模态独立的 Transformer 参数**，稀疏激活，互不干扰。

## 1. MoT 不是 MoE

这是最容易搞混的地方：

```
MoE (Mixture of Experts)
  一层 Transformer 内，FFN 拆成多个 expert
  学习的 Router 选 top-k expert 处理每个 token
  → 所有 token 共享 QKV 参数
  → 粒度：FFN 级

MoT (Mixture of Transformers)
  不同模态有独立的 Transformer block 参数
  （QKV 投影 + FFN + LayerNorm 全都独立）
  → 确定性路由：按 token 的 modality mask 直接分配 —— 不学习！
  → 粒度：Block 级
```

| | Dense Transformer | MoE | MoT |
|---|---|---|---|
| FFN 参数 | 共享 | 按 expert 拆分 | **按模态独立** |
| QKV 参数 | 共享 | 共享 | **按模态独立** |
| LayerNorm | 共享 | 共享 | **按模态独立** |
| 路由方式 | 无 | 学习的 Router | **确定性 modality mask** |
| 跨模态交互 | Self-attention | Self-attention | 共享 attention 或 Cross-attention |
| 参数效率 | 低（所有模态挤一套参数） | 中（FFN 专精，attention 共享） | **高（全栈专精）** |
| 代表工作 | GPT, LLaMA | Mixtral, Switch Transformer | **Meta MoT, Cosmos 3, HY-Embodied** |

**一句话：MoE 靠学习来选 expert，MoT 靠 token 是哪个模态来直接决定走哪个 Transformer。**

---

## 2. MoT 的起源：Meta (TMLR 2025)

> **Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models**
> Liang et al., Meta, TMLR 2025. [arXiv 2411.04996](https://arxiv.org/abs/2411.04996)

### 核心设计

```
        Token 序列（文本 + 图像 + 语音）
        │
        ├── token[0]: "The"      → modality_mask = text
        ├── token[1]: "cat"      → modality_mask = text
        ├── token[2]: <img_1>    → modality_mask = image
        ├── token[3]: <img_2>    → modality_mask = image
        └── token[4]: "sat"      → modality_mask = text

        ↓ 每一层 Transformer ↓

  ┌─────────────────────────────────────────┐
  │  Attention: 全局共享（跨模态交互）        │
  │  QKV 投影权重按 token 的 modality 选择    │
  │  → text token 用 text_qkv_weight         │
  │  → image token 用 image_qkv_weight       │
  └─────────────────────────────────────────┘
                    ↓
  ┌─────────────────────────────────────────┐
  │  FFN: 按 token 的 modality 选择           │
  │  → text token 用 text_ffn               │
  │  → image token 用 image_ffn             │
  └─────────────────────────────────────────┘
```

### 为什么有效？

```
直觉：文本 token "cat" 和图像 token <img_cat> 虽然语义相关，
但底层特征结构完全不同——一个关注语义，一个关注纹理/形状。

共享 QKV： 文本和图像的 query/key/value 在同一个空间 → 互相干扰
MoT QKV：  各自有独立的投影 → "各说各话"，在 attention 层交汇

实测：
  Chameleon 7B（文本生成 + 图像生成）
    Dense Transformer:   基线
    MoT:                 55.8% FLOPs，质量不降
  文本 + 图像 + 语音：
    MoT:                 仅用 37.2% FLOPs

消融实验（来自原论文）：
  FFN 解耦贡献最大 > QKV 解耦第二 > LayerNorm 解耦影响极小
  → 如果算力紧，至少把 FFN 分家
```

---

## 3. Cosmos 3：MoT 在物理世界模型中的实践（NVIDIA, 2026.06）

> **Cosmos 3: Omnimodal World Models for Physical AI**
> NVIDIA Cosmos Lab, 2026. [arXiv 2606.02800](https://arxiv.org/abs/2606.02800)

### 3.1 架构全景

```
Cosmos 3 = 双塔 MoT

  ┌─────────────────────┐     ┌─────────────────────┐
  │  AR Reasoner Tower   │     │  DM Generator Tower   │
  │  (理解/推理)          │────►│  (生成/合成)          │
  │                      │ 单向 │                      │
  │  Text Transformer    │ joint│  Video Transformer   │
  │  Image Transformer   │ attn │  Image Transformer   │
  │  Action Transformer  │      │  Audio Transformer   │
  └─────────────────────┘     └─────────────────────┘
           ↓                           ↓
    语义理解、推理              视频/图像/音频/动作 生成

关键：AR token 绝不接收 DM token 的更新 → 文本推理不受视觉噪声干扰

### 3.2 关键创新

**① 每个模态有自己的 Transformer 参数**

```
Reasoner:
  Text → TextTransformerBlock (QKV + FFN + LN)
  Image → ImageTransformerBlock (QKV + FFN + LN)
  Action → ActionTransformerBlock (QKV + FFN + LN)

Generator:
  Video → VideoTransformerBlock
  Image → ImageTransformerBlock
  Audio → AudioTransformerBlock
```

**② 双塔之间的单向联合注意力**

```
Reasoner 的中间特征 ──→ Generator 的 cross-attention（AR → DM）
Generator 的中间特征 ──✕── 不能更新 Reasoner（DM → AR 被阻断）

→ AR token 只从文本/图像/视频理解中学习，不被扩散过程的噪声污染
→ Generator 接收 Reasoner 的语义信息来引导生成
```

**③ 3D 多模态 RoPE**

```
视频 token 的时间轴 FPS₁
音频 token 的时间轴 FPS₂
动作 token 的时间轴 FPS₃

→ 3D RoPE 把三者对齐到统一的物理时间轴
→ "视频第 10 帧 = 音频 0.33 秒 = 动作第 100 步"
```

**④ 统一动作表示**

```
动作不是绝对控制信号，而是 "pseudo-action"：
  → 连续帧之间的状态差（Δ joint, Δ pose）
  → 与视频帧时间戳对齐

机械臂 Δ关节角 → shared latent geometric action space
自动驾驶 Δ转角  → shared latent geometric action space
人体姿态 Δ骨架  → shared latent geometric action space
```

### 3.3 模型规格

| | Edge | Nano | Super |
|---|---|---|---|
| 总参数 | 4B | 16B | 64B |
| 适用 | 端侧实时 | 中规模 | 云端 |

### 3.4 对具身的启示

```
Cosmos 3 的 MoT 设计对 robot policy 的意义：

1. Action Transformer 独立 → 动作 token 不走视觉/语言参数
   → 预测 SE(3) 位姿时不被高维视觉特征淹没

2. 单向双塔 → Reasoner 理解场景（不受视觉噪声污染），Generator 接收语义信息生成动作
   → Reasoner: "我看到一个杯子，应该抓它"（AR token 不会被 DM token 污染）
   → Generator: "抓杯子的轨迹是..."（接收 Reasoner 语义，迭代去噪生成轨迹）

3. 3D RoPE 的时间对齐 → 力触觉（1000Hz）、视觉（30Hz）、
   本体感觉（100Hz）可以对齐到统一时间轴
```

---

## 4. HY-Embodied-0.5：MoT 在具身模型中的落地（腾讯, 2026.04）

> **HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents**
> 腾讯 Robotics X + 混元, 2026. [arXiv 2604.07430](https://arxiv.org/abs/2604.07430)

### 4.1 架构

```
HY-Embodied MoT-2B:

  ┌──────────────────────────────────────┐
  │  LLM Backbone (文本处理)              │
  │  ┌──────────────────────────────────┐ │
  │  │ 视觉分支：独立的 FFN + QKV 层      │ │
  │  │ → 视觉 token 走独立的 Transformer  │ │
  │  │ → 双向注意力（视频理解更强）       │ │
  │  └──────────────────────────────────┘ │
  │  ┌──────────────────────────────────┐ │
  │  │ Visual Latent Tokens             │ │
  │  │ → 视觉序列末尾的潜在 token         │ │
  │  │ → 接受 ViT 全局特征监督            │ │
  │  │ → 作为"视觉-语言桥梁"              │ │
  │  └──────────────────────────────────┘ │
  └──────────────────────────────────────┘
```

### 4.2 关键数字

```
MoT-2B: 总参数 4B，激活仅 2.2B
  → 参数比同规模 Dense 模型多，但推理 FLOPs 接近

22 项具身评测：16 项最优（超过 Qwen3-VL 2B/4B、RoboBrain 2.5 4B）
```

### 4.3 为什么 MoT 对具身特别重要

```
传统 VLM 做 robot policy：
  视觉 token 占 80%+ 的序列长度
  → 视觉信息量虽大，但挤占了语言/动作的参数空间
  → "看图厉害，产生动作不精准"

MoT 做 robot policy：
  视觉有独立的 QKV + FFN → 不挤占动作参数
  语言有独立的 QKV + FFN → 指令理解不受视觉干扰
  动作 token 可以不经过视觉 FFN → "力控信号不被视觉稀释"

→ 这就是为什么 HY-Embodied MoT-2B 能超越 4B/7B 的 Dense 模型
```

---

## 5. MoT + MoE 的组合

MoT 和 MoE 不是互斥的——MoT 是框架，MoE 是里面的高效组件：

```
┌──────────────── MoT 框架 ────────────────┐
│                                          │
│  Vision Transformer Block                │
│    ├── Vision QKV (独立)                 │
│    ├── Vision Attn (可以双向)            │
│    └── Vision MoE FFN (MoE 在 FFN 里)    │  ← MoE!
│        Expert 1: 物体检测                │
│        Expert 2: 场景理解                │
│        Expert 3: 深度估计                │
│                                          │
│  Force/Tactile Transformer Block         │
│    ├── Force QKV (独立)                  │
│    ├── Force Attn                        │
│    └── Force MoE FFN                     │  ← MoE!
│        Expert 1: 法向力                  │
│        Expert 2: 切向力/滑移             │
│        Expert 3: 力矩                    │
│                                          │
│  Cross-Modal Attention                   │
│    Force Q · Vision K → 力感知对视觉     │
│    Vision Q · Force K → 视觉对力感知     │
│                                          │
└──────────────────────────────────────────┘
```

**MoT 提供模态隔离，MoE 在模态内部提供专家专精。** 两层稀疏：Block 级 + FFN 级。

---

## 6. MoT 设计空间

### 6.1 哪些模态有独立 Transformer？

| 策略 | 做法 | FLOPs | 适用 |
|------|------|-------|------|
| 最小 MoT | 只有视觉独立 | ~70% Dense | 视觉主导任务 |
| 标准 MoT | 视觉 + 语言独立 | ~60% Dense | VLA 通用 |
| **完整 MoT** | **视觉 + 语言 + 力触觉 + 动作 全独立** | ~40-50% Dense | **你的设计** |

### 6.2 跨模态交互如何做？

```
选项 A：共享 Self-Attention（Meta MoT 原版）
  → 所有 token 在注意力层见面
  → QKV 投影按 modality 独立，但 attention score 全局计算

选项 B：Cross-Attention（Cosmos 3 双塔）
  → 每个模态独立做 self-attention
  → 模态间通过 cross-attention 交互

选项 C：Token Exchange（稀疏交互）
  → 只在特定层交换少量"摘要 token"
  → 最省算力
```

### 6.3 频率对齐

MoT 天然解决频率问题——因为各模态独立：

```
视觉 30Hz    → Vision Transformer (30 fps 编码)
力觉 1000Hz  → Force Transformer (1000Hz 高频编码，不需要降采样！)
触觉 60Hz    → Tactile Transformer (60Hz 编码)
动作 10Hz    → Action Transformer (10Hz 解码)
```

---

## 7. 对你的架构设计意味着什么

```
你的 MoT 方案（面试答案骨架）：

① 模态分家：
   Vision Expert → 独立的 Visual Transformer Block
   Force Expert  → 独立的 F/T Transformer Block
   Tactile Expert → 独立的 Tactile Transformer Block
   Language → 预训练 LLM block（冻结或微调）
   Action Head → FM Head（不变）

② 交互层：
   每隔 N 层插入 Cross-Attention：
     Force Q · Vision K → "视觉上看到的物体位置 → 引导力方向"
     Vision Q · Force KV → "力反馈验证视觉判断 → 确认抓稳了"

③ MoE 内嵌：
   力觉分支内部用 MoE：
     Force-Normal Expert（法向力控制）
     Force-Tangential Expert（切向力/滑移检测）
     Force-Moment Expert（力矩平衡）

④ MoT Router = 模态 + 阶段联合路由：
   Router(vision_token, force_token, phase_embedding) → 
     预接触：90% Visual Expert, 10% Force Expert
     接触瞬间：50% Visual, 50% Force
     操作中：30% Visual, 70% Force/Tactile
```

---

## 📚 学习资源

### 📄 本地论文（项目内可直接打开）

| 论文 | 路径 |
|------|------|
| Meta MoT (TMLR 2025) | [`docs/Papers/Mixture-of-Transformers-Meta-TMLR2025.pdf`](../Papers/Mixture-of-Transformers-Meta-TMLR2025.pdf) |
| Cosmos 3 (2026.06) | [`docs/Papers/Cosmos3-Omnimodal-World-Models-2026.pdf`](../Papers/Cosmos3-Omnimodal-World-Models-2026.pdf) |

### 🌐 在线资料

| 资源 | 说明 |
|------|------|
| 📝 [NVIDIA Cosmos 3 Technical Blog](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/) | Cosmos 3 架构详解（官方） |
| 📝 [NVIDIA MoT Glossary](https://www.nvidia.com/en-us/glossary/mixture-of-transformers/) | MoT 官方定义 |
| 📝 [HuggingFace: Cosmos 3 for Physical AI](https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai) | HF 官方博客 |
| 📄 [MoE 博客 (HuggingFace)](https://huggingface.co/blog/moe) | MoE 基础（MoT 的前置知识） |
| 🎥 [Yannic Kilcher: Switch Transformer](https://www.youtube.com/watch?v=iAR8LkkMMIM) | MoE 论文精读 |
| 💻 [Meta MoT 代码](https://github.com/facebookresearch/Mixture-of-Transformers) | 官方实现 |
| 💻 [Cosmos 3 代码](https://github.com/nvidia/cosmos) | NVIDIA 官方实现 |
| 💻 [HY-Embodied-0.5 模型](https://huggingface.co/AndyYehoo/HY-Embodied-0.5) | 具身 MoT 落地 |

### 先读 3 篇

1. ⭐⭐⭐ Liang et al., "Mixture-of-Transformers: A Sparse and Scalable Architecture for Multi-Modal Foundation Models", TMLR 2025 — **MoT 起源** → [`PDF`](../Papers/Mixture-of-Transformers-Meta-TMLR2025.pdf) · [`arXiv`](https://arxiv.org/abs/2411.04996)
2. ⭐⭐⭐ NVIDIA Cosmos Lab, "Cosmos 3: Omnimodal World Models for Physical AI", 2026.06 — **MoT 在物理世界模型中的最新实践** → [`PDF`](../Papers/Cosmos3-Omnimodal-World-Models-2026.pdf) · [`arXiv`](https://arxiv.org/abs/2606.02800) · [`技术博客`](https://developer.nvidia.com/blog/develop-physical-ai-reasoning-world-and-action-models-with-nvidia-cosmos-3/)
3. ⭐⭐ Tencent, "HY-Embodied-0.5: Embodied Foundation Models for Real-World Agents", 2026.04 — **MoT 在具身模型中最接近落地的案例** → [`arXiv`](https://arxiv.org/abs/2604.07430) · [`模型`](https://huggingface.co/AndyYehoo/HY-Embodied-0.5)

---

## ✍️ 写作练习

### L1 费曼笔记

- [ ] 用大白话解释：**MoT 和 MoE 到底有什么区别？为什么 MoT 特别适合多模态机器人？**

### L2 概念串联

- [ ] 画出你的 MoT 架构图：哪些模态有独立 Transformer？Cross-attention 在哪些层？MoE 嵌在哪里？Router 怎么根据接触阶段切换？

### L3 设计性写作

→ 写到 [I4-累积设计文档](/robotics-learning-map/I4-累积设计文档) Section 3：你的多模态 MoT 融合架构

---

## 🚫 常见弯路

- ❌ 把 MoT 当成 MoE 来学 → MoT 是 Block 级分家，MoE 是 FFN 级分家
- ❌ 每个模态都独立 Transformer 但不要 cross-attention → 模态完全隔离等于"各干各的"
- ❌ 所有模态都给独立 QKV → 力觉 token 只有 1-2 个，独立 QKV 性价比低（用共享 QKV + 独立 FFN）

---

## 🔗 相关笔记

- [M1-Riemannian-FM与SE3生成](/robotics-learning-map/M1-Riemannian-FM与SE3生成) — FM Head 接在 Transformer 后面
- [M3-触觉表征与力预测](/robotics-learning-map/M3-触觉表征与力预测) — 触觉/力 encoder 设计
- [I4-累积设计文档](/robotics-learning-map/I4-累积设计文档)
- [写作工具箱](/robotics-learning-map/写作工具箱)
- 📄 下载: `docs/Papers/Mixture-of-Transformers-Meta-TMLR2025.pdf`
- 📄 下载: `docs/Papers/Cosmos3-Omnimodal-World-Models-2026.pdf`
