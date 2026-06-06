/**
 * VitePress markdown-it 插件：渲染时将 [[wiki-links]] 转换为标准链接。
 * 不修改原始 .md 文件——你可以在 VS Code 里继续用 [[双向链接]] 写笔记。
 */

import type MarkdownIt from 'markdown-it'
import { readdirSync } from 'node:fs'
import { join } from 'node:path'

// 扫描一次，建立已知文档名集合
const docsDir = join(import.meta.dirname, '..', '..', 'docs')
let knownDocs: Set<string> | null = null

function getKnownDocs(): Set<string> {
  if (knownDocs) return knownDocs
  try {
    const files = readdirSync(docsDir).filter(f => f.endsWith('.md'))
    knownDocs = new Set(files.map(f => f.replace(/\.md$/, '')))
  } catch {
    knownDocs = new Set()
  }
  return knownDocs
}

export function wikiLinkPlugin(md: MarkdownIt) {
  const docs = getKnownDocs()

  // 在 inline 解析前处理 [[links]]
  md.inline.ruler.before('link', 'wiki_link', (state, silent) => {
    const src = state.src
    const pos = state.pos

    if (src[pos] !== '[' || src[pos + 1] !== '[') return false

    const match = src.slice(pos).match(/^\[\[([^\]|#]+)(?:#([^\]|]*))?(?:\|([^\]]+))?\]\]/)
    if (!match) return false

    if (silent) return true

    const [, docName, anchor, alias] = match
    const display = alias || docName
    const anchorPart = anchor ? '#' + anchor.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9一-鿿\-]/g, '') : ''

    if (!docs.has(docName)) {
      // 未知文档——渲染为斜体文字
      const token = state.push('em_open', 'em', 1)
      token.markup = '*'
      const textToken = state.push('text', '', 0)
      textToken.content = display
      state.push('em_close', 'em', -1)
      state.pos = pos + match[0].length
      return true
    }

    const url = `/${docName}${anchorPart}`

    // 打开 <a>
    const linkOpen = state.push('link_open', 'a', 1)
    linkOpen.attrs = [['href', url]]

    // 链接文字
    const textToken = state.push('text', '', 0)
    textToken.content = display

    // 关闭 </a>
    state.push('link_close', 'a', -1)

    state.pos = pos + match[0].length
    return true
  })
}
