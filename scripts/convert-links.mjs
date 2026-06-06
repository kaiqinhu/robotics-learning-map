/**
 * 把 Obsidian 风格的 [[wiki-links]] 转换成 VitePress 兼容的 Markdown 链接。
 * 每次 build 前运行。
 *
 * [[文档名]]             → [文档名](/文档名)
 * [[文档名#section]]     → [文档名](/文档名#section)
 * [[文档名|显示文字]]     → [显示文字](/文档名)
 * [[文档名#sec|文字]]    → [文字](/文档名#sec)
 */

import { readdir, readFile, writeFile } from 'node:fs/promises'
import { join, extname } from 'node:path'

const DOCS_DIR = join(import.meta.dirname, '..', 'docs')

// Gather all known doc names (without .md)
const files = await readdir(DOCS_DIR)
const knownDocs = new Set(
  files
    .filter(f => extname(f) === '.md')
    .map(f => f.replace(/\.md$/, ''))
)

// Also gather known section anchors from each file
// We'll just match [[filename]] and [[filename#anything]]
const wikiLinkRe = /\[\[([^\]|#]+)(?:#([^\]|]*))?(?:\|([^\]]+))?\]\]/g

const mdFiles = files.filter(f => extname(f) === '.md')

for (const file of mdFiles) {
  const filepath = join(DOCS_DIR, file)
  let content = await readFile(filepath, 'utf-8')
  let modified = false

  content = content.replace(wikiLinkRe, (match, docName, anchor, alias) => {
    // Check if the linked document exists
    if (!knownDocs.has(docName)) {
      console.warn(`  ⚠ [${file}] 链接到未知文档: [[${docName}]]`)
      // Keep the link but it'll be broken — user needs to fix it
      const display = alias || docName
      return `*${display}*`
    }

    const display = alias || docName
    const anchorPart = anchor ? `#${anchor.toLowerCase().replace(/\s+/g, '-')}` : ''
    const url = `/robotics-learning-map/${docName}${anchorPart}`
    modified = true
    return `[${display}](${url})`
  })

  if (modified) {
    await writeFile(filepath, content, 'utf-8')
    console.log(`  ✓ ${file}`)
  }

  // Second pass: fix hardcoded absolute internal links like [text](/docName) → [text](/base/docName)
  const absoluteLinkRe = /\[([^\]]*)\]\(\/([^)\s]+)\)/g
  content = content.replace(absoluteLinkRe, (match, text, path) => {
    // Skip external URLs (shouldn't match since they start with http, but be safe)
    if (path.startsWith('http')) return match

    const [docName, anchor] = path.split('#')
    const anchorSuffix = anchor ? `#${anchor}` : ''

    // Root link "/"
    if (docName === '') {
      modified = true
      return `[${text}](/robotics-learning-map/${anchorSuffix})`
    }

    // Known internal doc
    if (knownDocs.has(docName)) {
      modified = true
      return `[${text}](/robotics-learning-map/${docName}${anchorSuffix})`
    }

    return match
  })

  if (modified) {
    await writeFile(filepath, content, 'utf-8')
    console.log(`  ✓ ${file} (internal links fixed)`)
  }
}

console.log('  → 链接转换完成')
