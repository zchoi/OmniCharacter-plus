# OmniCharacter++ — Project Page

Static project page for **OmniCharacter++: Towards Comprehensive Benchmark for Realistic Role-Playing Agents**. Plain HTML / CSS / vanilla JS, no build step, hosted on GitHub Pages.

🔗 **Live**: <https://zchoi.github.io/OmniCharacter-plus/>

## 页面内容

```
Title + Authors + Buttons  ─►  Headline Results  ─►  Model Framework  ─►  CharacterEval Generalization  ─►  Abstract  ─►  BibTeX
```

页面打开后直接呈现论文标题、作者和入口按钮，随后展示核心仿真实验、真机表格和已剪辑视频；措辞按论文正文和表格收紧，避免把分数扩展成更强的结论。

| 段落 | 内容 |
|------|------|
| **Title + Authors + Buttons** | 论文标题、网站展示作者和机构；`OmniCharacter++` 用 Cornell 红高亮；Paper / Code / Models / Framework / BibTeX 可点击 |
| **Headline Results** | `intro.png` overview 大图 + 4 张数字卡（**10K+** characters / **118K+** dialogues / **1M+** speeches / **3941.76h** speech）+ Multi-party Context Understanding HTML 表格；其它实验提示见 paper |
| **Model Framework** | `framework.png` 展示 OmniCharacter++ speech-language collaborative model、emotion preference learning 和 role-contextual dialogue adaptation |
| **CharacterEval Generalization** | CharacterEval 泛化性测试 HTML 表格，绿色标 OmniCharacter-7B，红色标 UniCharacter-7B |
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
    ├── js/main.js                      # BibTeX 复制
    ├── videos/                         # 真机视频，来自 ref/video-edited
    │   ├── hexagon_*.mp4               # 4 段：original + 3 OOD
    │   ├── zipper_*.mp4                # 4 段
    │   ├── vase_*.mp4                  # 4 段
    │   └── bottle_*.mp4                # 4 段
    └── images/
        ├── teaser.png                  # 用：社交分享 OG image
        ├── intro.png                   # 用：OmniCharacter++ overview
        ├── framework.png               # 用：model framework
        ├── data_dis.png                # 备用：topic / scenario distribution
        ├── len.png                     # 备用：dialogue length / audio duration distribution
        ├── dyadic_conv.png             # 备用：dyadic context understanding results
        └── multi_conv.png              # 备用：multi-party context understanding screenshot
```

## 论文实验数据

### OmniCharacter++ Benchmark

| Set | Dialogue Type | #Characters | Avg. Turns/Conv. | #Dialogues | #Speech Hours |
|-----|---------------|-------------|------------------|------------|---------------|
| Train | Dyadic | 10,277 | 10.00 | 88,474 | 2867.94 |
| Train | Multi-Party | 10,277 | 15.05 | 29,543 | 1051.66 |
| Test | Dyadic | 10 | 9.89 | 185 | 6.96 |
| Test | Multi-Party | 10 | 16.72 | 334 | 15.20 |
| **Total** | - | **10,377** | **12.92** | **118,536** | **3941.76** |

### Model Framework

`static/images/framework.png` 展示 OmniCharacter++ 的 speech-language collaborative model，包括 role-aware speech decoder、emotion preference learning 和 role-contextual dialogue adaptation。

## 本地预览

```bash
python3 -m http.server 8000
# 打开 http://localhost:8000
```

## 待正式发布后补

| 位置 | 替换什么 |
|------|---------|
| `#bibtex` 内 `<pre><code>` 块 | Google Scholar BibTeX 条目 |

## 模型框架图片

页面使用 `static/images/framework.png`，来源于仓库根目录 `assets/framework.png`。

## 自定义

### 主题色

`static/css/style.css` 顶部 `:root`：

```css
--accent: #b11f3a;        /* Cornell 红 — 标题 OmniCharacter++ / 高亮行 / callout 边线 */
--accent-blue: #2f5f8f;   /* baseline 蓝 — 学习曲线注释 */
```

## Tech notes

- **KaTeX** via CDN（jsdelivr）：所有 `$...$` / `$$...$$` 自动渲染。当前页面 KaTeX 主要用于公式，CDN 保留以备将来使用。
- **Font Awesome 6** via CDN：所有按钮图标
- **Google Fonts** Inter (sans) + Noto Serif (标题)

## 致谢

布局借鉴 [Nerfies](https://github.com/nerfies/nerfies.github.io)、[ManualVLA](https://sites.google.com/view/maunalvla/) 和 [LaST₀](https://vla-last0.github.io/)。
