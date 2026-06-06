<script setup>
import DefaultTheme from 'vitepress/theme'
import { ref, onMounted } from 'vue'

const { Layout } = DefaultTheme

const hideS = ref(false)
const hideA = ref(false)

const STORAGE_S = 'vp-hide-sidebar'
const STORAGE_A = 'vp-hide-aside'

onMounted(() => {
  hideS.value = document.documentElement.classList.contains('hide-sidebar')
  hideA.value = document.documentElement.classList.contains('hide-aside')
})

function update() {
  localStorage.setItem(STORAGE_S, hideS.value)
  localStorage.setItem(STORAGE_A, hideA.value)
  document.documentElement.classList.toggle('hide-sidebar', hideS.value)
  document.documentElement.classList.toggle('hide-aside', hideA.value)
}
</script>

<template>
  <Layout>
    <!-- 左侧边栏顶部 -->
    <template #sidebar-nav-before>
      <div class="sb-top">
        <span class="sb-label">导航</span>
        <button class="x-btn" title="收起侧边栏" @click="hideS = true; update()">✕</button>
      </div>
    </template>

    <!-- 右侧目录顶部 -->
    <template #aside-top>
      <div class="sb-top">
        <span class="sb-label">本页目录</span>
        <button class="x-btn" title="收起目录" @click="hideA = true; update()">✕</button>
      </div>
    </template>
  </Layout>

  <!-- 边缘恢复 tab：收起后才出现 -->
  <Teleport to="body">
    <div v-show="hideS" class="edge edge-left" title="展开侧边栏" @click="hideS = false; update()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
      <span>目录</span>
    </div>
    <div v-show="hideA" class="edge edge-right" title="展开目录" @click="hideA = false; update()">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="15 18 9 12 15 6"/></svg>
      <span>目录</span>
    </div>
  </Teleport>
</template>

<style>
/* ===== 顶部行（侧边栏 & 目录共用） ===== */
.sb-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 16px 8px 16px;
}
.VPDocAside .sb-top {
  padding: 0 0 8px 0;
}
.sb-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--vp-c-text-2);
}

/* ===== 收起 ✕ 按钮 ===== */
.x-btn {
  width: 22px; height: 22px;
  display: inline-flex;
  align-items: center; justify-content: center;
  border: none; border-radius: 4px;
  background: transparent;
  color: var(--vp-c-text-3);
  cursor: pointer; font-size: 11px;
  transition: all 0.15s;
  flex-shrink: 0;
}
.x-btn:hover {
  background: var(--vp-c-bg-mute);
  color: var(--vp-c-text-1);
}

/* ===== 边缘恢复 tab ===== */
.edge {
  position: fixed;
  top: 50%;
  transform: translateY(-50%);
  z-index: 998;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 3px;
  width: 28px;
  padding: 14px 4px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-3);
  cursor: pointer;
  user-select: none;
  transition: all 0.2s;
}
.edge:hover {
  color: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  background: var(--vp-c-bg-mute);
}
.edge svg {
  flex-shrink: 0;
}
.edge span {
  writing-mode: vertical-rl;
  font-size: 10px;
  letter-spacing: 2px;
}
.edge-left {
  left: 0;
  border-left: none;
  border-radius: 0 8px 8px 0;
}
.edge-right {
  right: 0;
  border-right: none;
  border-radius: 8px 0 0 8px;
}

/* ===== 收起/展开 CSS ===== */
html.hide-sidebar .VPSidebar { display: none !important; }
html.hide-sidebar .VPContent.has-sidebar { padding-left: 0 !important; }
html.hide-aside .VPDoc.has-aside .aside { display: none !important; }
html.hide-aside .VPDoc.has-aside .container { margin-right: 0 !important; }

.VPSidebar, .VPContent.has-sidebar, .VPDoc.has-aside .aside {
  transition: all 0.25s ease;
}
</style>
