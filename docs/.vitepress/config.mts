import { defineConfig } from 'vitepress'

// 防 FOUC：在任何 DOM 渲染之前读取 localStorage 并应用 class
const antiFouc = `
(function() {
  try {
    if (localStorage.getItem('vp-hide-sidebar') === 'true') {
      document.documentElement.classList.add('hide-sidebar');
    }
    if (localStorage.getItem('vp-hide-aside') === 'true') {
      document.documentElement.classList.add('hide-aside');
    }
  } catch(e) {}
})();
`

export default defineConfig({
  title: '具身操作基础模型 · 学习地图',
  description: 'Flow Matching + MoT + SE(3) + 力触觉 + 灵巧手 — 从基础到系统',
  lang: 'zh-CN',
  base: '/robotics-learning-map/',
  ignoreDeadLinks: true,

  head: [['script', {}, antiFouc]],

  themeConfig: {
    search: {
      provider: 'local',
    },

    nav: [
      { text: '首页', link: '/' },
      {
        text: '🔴 Core',
        items: [
          { text: 'C0 · 动作生成范式全景', link: '/C0-动作生成范式全景' },
          { text: 'C1 · 经典机器人学', link: '/C1-经典机器人学' },
          { text: 'C2 · SE(3) 几何入门', link: '/C2-SE3几何入门' },
          { text: 'C3 · Flow Matching 基础', link: '/C3-Flow-Matching基础' },
          { text: 'C4 · 阻抗与导纳控制', link: '/C4-阻抗与导纳控制' },
        ]
      },
      {
        text: '🟡 Methods',
        items: [
          { text: 'M1 · Riemannian FM 与 SE(3)', link: '/M1-Riemannian-FM与SE3生成' },
          { text: 'M2 · 多模态 MoT', link: '/M2-多模态MoT' },
          { text: 'M3 · 触觉表征与力预测', link: '/M3-触觉表征与力预测' },
          { text: 'M4 · 预训练范式', link: '/M4-预训练范式' },
          { text: 'M5 · 后训练与持续学习', link: '/M5-后训练与持续学习' },
        ]
      },
      {
        text: '🟢 Integration',
        items: [
          { text: 'I1 · 系统架构与部署', link: '/I1-系统架构与部署' },
          { text: 'I2 · 数据工程', link: '/I2-数据工程' },
          { text: 'I3 · 评估与基准', link: '/I3-评估与基准' },
          { text: 'I4 · 累积设计文档', link: '/I4-累积设计文档' },
        ]
      },
      { text: '📡 最新进展', link: '/最新进展-2025-2026' },
    ],

    sidebar: [
      { text: '🗺️ 入口', items: [
        { text: '总览·学习地图', link: '/' },
      ]},
      { text: '🔴 Core 核心层', collapsed: false, items: [
        { text: 'C0 动作生成范式全景', link: '/C0-动作生成范式全景' },
        { text: 'C1 经典机器人学', link: '/C1-经典机器人学' },
        { text: 'C2 SE(3)几何入门', link: '/C2-SE3几何入门' },
        { text: 'C3 Flow Matching基础', link: '/C3-Flow-Matching基础' },
        { text: 'C4 阻抗与导纳控制', link: '/C4-阻抗与导纳控制' },
      ]},
      { text: '🟡 Methods 方法层', collapsed: false, items: [
        { text: 'M1 Riemannian FM与SE3生成', link: '/M1-Riemannian-FM与SE3生成' },
        { text: 'M2 多模态MoT', link: '/M2-多模态MoT' },
        { text: 'M3 触觉表征与力预测', link: '/M3-触觉表征与力预测' },
        { text: 'M4 预训练范式', link: '/M4-预训练范式' },
        { text: 'M5 后训练与持续学习', link: '/M5-后训练与持续学习' },
      ]},
      { text: '🟢 Integration 集成层', collapsed: false, items: [
        { text: 'I1 系统架构与部署', link: '/I1-系统架构与部署' },
        { text: 'I2 数据工程', link: '/I2-数据工程' },
        { text: 'I3 评估与基准', link: '/I3-评估与基准' },
        { text: 'I4 累积设计文档', link: '/I4-累积设计文档' },
      ]},
      { text: '📡 前沿与交流', collapsed: false, items: [
        { text: '最新进展 2025-2026', link: '/最新进展-2025-2026' },
        { text: '交流记录与洞察', link: '/10-交流记录与洞察' },
      ]},
      { text: '', items: [
        { text: '写作工具箱', link: '/写作工具箱' },
        { text: '更新日志', link: '/changelog' },
        { text: '许可与联系', link: '/license' },
      ]},
    ],

    outline: { level: [2, 3], label: '本页目录' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    darkModeSwitchLabel: '深色模式',
    sidebarMenuLabel: '菜单',
    returnToTopLabel: '回到顶部',
  },
})
