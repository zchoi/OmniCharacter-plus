# LaST-R1 — Project Page

Static project page for **LaST-R1: Reinforcing Robotic Manipulation via Adaptive Physical Latent Reasoning**. Plain HTML / CSS / vanilla JS, no build step, hosted on GitHub Pages.

🔗 **Live**: <https://siriyep.github.io/last-r1/>

## 页面内容

```
Teaser  ─►  Title + Buttons  ─►  Headline Results (Sim)  ─►  Real-World  ─►  Abstract  ─►  BibTeX
```

页面直接呈现论文主图、核心仿真实验、真机表格和已剪辑视频；措辞按论文正文和表格收紧，避免把分数扩展成更强的结论。

| 段落 | 内容 |
|------|------|
| **Hero intro** | 进页面瞬间全屏显示标题 + 2×2 真机视频墙，第一次滚动触发 FLIP morph 动画把标题滑到正文位置；hash 直链（`#bibtex` 等）和 `prefers-reduced-motion` 用户跳过此动画 |
| **Teaser** | 论文 Figure 1 (`teaser.png`) |
| **Title + Authors + Buttons** | 论文标题、网站展示作者和机构；`LaST-R1` 用 Cornell 红高亮；Paper / Code / Models / Videos / BibTeX 可点击 |
| **Headline Results** | 4 张数字卡（**99.9%** LIBERO / **+22.5%** real-world over SOTA SFT / **1 traj** warm-up / sim+real generalization）+ 完整 LIBERO 12 行对比表（`tab:libero_comparison`）+ 最新学习曲线图（`main_results.png`）+ generalization 图（`main_ablation_gen.png`）+ 三联 callout |
| **Real-World** | 论文 Table 2 真机成功率（`π0.5` Full-size SFT vs. LaST-R1 Few-shot SFT→RL）+ `video-edited` 里的 16 段真机视频 |
| **Abstract** | 论文 abstract 原文 |
| **BibTeX** | 引用块 + 一键复制按钮 |

## 目录结构

```
.
├── index.html                          # 整个页面（约 730 行）
├── README.md
├── .gitignore
├── .nojekyll                           # 让 GitHub Pages 跳过 Jekyll
└── static/
    ├── css/style.css                   # 所有样式
    ├── js/main.js                      # intro morph 逻辑 + BibTeX 复制
    ├── videos/                         # 真机视频，来自 ref/video-edited
    │   ├── hexagon_*.mp4               # 4 段：original + 3 OOD
    │   ├── zipper_*.mp4                # 4 段
    │   ├── vase_*.mp4                  # 4 段
    │   └── bottle_*.mp4                # 4 段
    └── images/
        ├── teaser.png                  # 用：teaser + 社交分享 OG image
        ├── main_results.png            # 用：学习曲线
        └── main_ablation_gen.png       # 用：LIBERO generalization 图
```

## 论文实验数据

### LIBERO Benchmark（仿真）
| Suite | LaST-R1 | π_RL（次优） |
|-------|---------|--------------|
| Spatial | **99.8** | 99.6 |
| Object | **100.0** | 100.0 |
| Goal | **100.0** | 99.6 |
| Long | **99.8** | 94.0 |
| **Average** | **99.9** | 98.3 |

### Real-World 真机（4 任务，SOTA SFT vs Few-shot SFT→RL，含 OOD 三列）

每任务格式：`Original / Unseen-Object / -Background / -Lighting`

| Task | π0.5 Full-size SFT | LaST-R1 Few-shot SFT→RL |
|------|--------------------|--------------------------|
| Insert hexagon block (single) | 65 / 35 / 55 / 40 | **45→90 / 75 / 85 / 80** |
| Open bag zipper (dual) | 75 / 30 / 70 / 60 | **55→95 / 80 / 95 / 90** |
| Wipe vase with sponge (dual) | 75 / 45 / 65 / 50 | **65→95 / 80 / 90 / 95** |
| Open bottle cap (dual) | 70 / 50 / 55 / 55 | **45→95 / 95 / 80 / 85** |
| **Original 列平均** | 71.25 | **52.5→93.75** |

## 本地预览

```bash
python3 -m http.server 8000
# 打开 http://localhost:8000
```

## 待正式发布后补

| 位置 | 替换什么 |
|------|---------|
| `#bibtex` 内 `<pre><code>` 块 | Google Scholar BibTeX 条目 |

## 真机视频

页面现在使用 `ref/video-edited/` 里的 16 段已剪辑视频，HTML 里 hardcode 了部署文件名（**不要改名**）。

| 任务 | 槽位 | 文件名 |
|------|------|--------|
| **Insert hexagon block** (single-arm) | Original | `static/videos/hexagon_original.mp4` |
| | Unseen-Object | `static/videos/hexagon_object.mp4` |
| | Unseen-Background | `static/videos/hexagon_background.mp4` |
| | Unseen-Lighting | `static/videos/hexagon_lighting.mp4` |
| **Open bag zipper** (dual-arm) | Original | `static/videos/zipper_original.mp4` |
| | Unseen-Object | `static/videos/zipper_object.mp4` |
| | Unseen-Background | `static/videos/zipper_background.mp4` |
| | Unseen-Lighting | `static/videos/zipper_lighting.mp4` |
| **Wipe vase with sponge** (dual-arm) | Original | `static/videos/vase_original.mp4` |
| | Unseen-Object | `static/videos/vase_object.mp4` |
| | Unseen-Background | `static/videos/vase_background.mp4` |
| | Unseen-Lighting | `static/videos/vase_lighting.mp4` |
| **Open bottle cap** (dual-arm) | Original | `static/videos/bottle_original.mp4` |
| | Unseen-Object | `static/videos/bottle_object.mp4` |
| | Unseen-Background | `static/videos/bottle_background.mp4` |
| | Unseen-Lighting | `static/videos/bottle_lighting.mp4` |
**视频建议规格**：H.264 / 720p+ / 16:9 / 单文件 < 15 MB。压缩命令（需 `ffmpeg`）：
```bash
ffmpeg -i in.mp4 -vcodec libx264 -crf 28 -preset slow -an out.mp4
```

GitHub 单文件 100 MB 上限，仓库总大小 1 GB 上限。

## 自定义

### 主题色

`static/css/style.css` 顶部 `:root`：

```css
--accent: #b11f3a;        /* Cornell 红 — 标题 LaST-R1 / LaST-R1 表格行 / callout 边线 */
--accent-blue: #2f5f8f;   /* baseline 蓝 — 学习曲线注释 */
```

### Intro 动画时长

`static/js/main.js` 顶部：

```js
var MORPH_MS = 950;   // 标题 / 图片 morph 持续时间
var FADE_MS  = 350;   // morph 完成后 veil 整体淡出时间
```

### 跳过 intro

加 `?` 后的任意 `#hash`（比如 `#bibtex`），或在系统设置里打开"减少动效"。也可以注释掉 `<head>` 里加 `intro-active` class 的那段 `<script>`。

## Tech notes

- **KaTeX** via CDN（jsdelivr）：所有 `$...$` / `$$...$$` 自动渲染。当前页面 KaTeX 主要用于公式，CDN 保留以备将来使用。
- **Font Awesome 6** via CDN：所有按钮图标
- **Google Fonts** Inter (sans) + Noto Serif (标题)
- **Intro morph**：FLIP 思路，JS 同步 veil 标题和 in-flow 标题的排版 metrics，再加 `translate(dx,dy)` → CSS `transition: transform 0.95s` 驱动平滑变换 → veil 淡出。开场 2×2 视频墙只在 veil 中展示，不参与 morph。

## 致谢

布局借鉴 [Nerfies](https://github.com/nerfies/nerfies.github.io)、[ManualVLA](https://sites.google.com/view/maunalvla/) 和 [LaST₀](https://vla-last0.github.io/)。
