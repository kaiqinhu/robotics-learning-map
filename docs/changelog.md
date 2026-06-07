# 更新日志

## v2.1 — 2026-06-07

**所有模块增强开头，三段式：背景 → 学到什么 → 驱动问题**

- 12 个核心模块（C0-C4, M1-M5, I1-I3）新增详细"背景"段落
- 每个模块列出 5 项具体可验证的学习目标
- 每个模块配 5 个核心驱动问题（面试级深度）
- 共计 414 行新增内容

## v2.0 — 2026-06-07

**全模块核查修订 + 论文引用体系建立**

### 修正
- **M2 多模态 MoT**：MoT 路由方式（确定性模态 mask vs 学习 Router）；Cosmos 3 注意力方向（双向→单向 AR→DM）；新增 pseudo-action 说明；HY-Embodied 描述精确化
- **M3 触觉表征**：OmniVLA 标注（原论文用雷达/红外而非力触觉）；HTD 日期修正（2025→2026.04）
- **M5 后训练**：ReinFlow wall time 澄清（23% 操作 / 83% 腿式运动）；FLaRe vs FLARE 区分标注
- **C0 动作生成范式**：FlowRAM +12% 是高精度子任务非全局
- **M4 预训练范式**：SPEAR-1 匹敌对象精确化（π₀-FAST 和 π₀.₅）
- **I3 评估基准**：大修——RLBench 已淘汰；新增 LIBERO-Plus / RoboCasa / RoboChallenge 最新排行；DobotWAM

### 新增
- **I1 系统架构**：Fast-WAM（测试时不需要未来想象）；Flow to One Step、Streaming Flow Policy、GPC 确认真实
- **参考资料索引** `docs/references/README.md`：完整引用索引
- 所有核查模块挂本地 PDF 链接 + 原始 URL

### 论文库
- 19 篇论文入 `docs/Papers/`（全部 gitignored 不入库）
- 涵盖 C0/M2/M3/M4/M5/I1

## v1.1 — 2026-06-07

**网站基础设施完善**

- GitHub Pages base path 修复（`/robotics-learning-map/`）
- 链接转换脚本增加硬编码绝对链接修复 pass
- 新增许可与联系页面（CC BY-NC-SA 4.0）
- 防爬邮箱：mongxianghu AT qq DOT com

## v1.0 — 2026-06-06

**初始发布**

- 三层螺旋学习框架：Core → Methods → Integration
- 18 篇学习文档 + 写作工具箱
- VitePress 静态站：可收起侧边栏、深色模式、本地搜索
- GitHub Actions 自动部署到 GitHub Pages

### 内容清单
- C0 动作生成范式全景（自回归 / 扩散 / FM / 掩码生成）
- C1 经典机器人学
- C2 SE(3) 几何入门
- C3 Flow Matching 基础
- C4 阻抗与导纳控制
- M1 Riemannian FM 与 SE(3) 生成
- M2 多模态 MoT
- M3 触觉表征与力预测（含六种融合范式 + 面试回答模板）
- M4 预训练范式
- M5 后训练与持续学习
- I1 系统架构与部署
- I2 数据工程
- I3 评估与基准
- I4 累积设计文档
- 最新进展 2025-2026
- 交流记录与洞察
- 写作工具箱

---

[回首页](/robotics-learning-map/)
