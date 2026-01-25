# 高级图表模板库

> **论文级专业图表模板** - 符合 Nature、Science 等顶级期刊规范，专为数学建模竞赛设计

## 📊 模板列表

### 1. 科学出版物级图表 (`scientific_plots.py`)

提供符合顶级期刊规范的图表模板：

- **Nature 期刊风格**: 符合 Nature 系列期刊的图表规范
- **Science 期刊风格**: 符合 Science 期刊的图表规范
- **IEEE 期刊风格**: 符合 IEEE 期刊的图表规范
- **多面板图表**: 支持创建复杂的子图布局
- **带注释图表**: 突出显示关键数据点

**特点**:
- ✅ 标准期刊尺寸（单栏、双栏）
- ✅ 高分辨率输出（600 DPI）
- ✅ 专业字体和字号
- ✅ 精确的线条宽度
- ✅ 符合可访问性要求

**使用示例**:
```python
from visualization.advanced_templates.scientific_plots import plot_publication_figure
import numpy as np

x = np.linspace(0, 10, 50)
y = np.sin(x)

# 创建 Nature 风格图表
fig, ax = plot_publication_figure(
    x, y,
    xlabel='Time (s)',
    ylabel='Amplitude',
    style='nature',
    save_path='figure1.png',
    dpi=600
)
```

### 2. 热力图模板 (`heatmaps.py`)

提供多种热力图可视化：

- **基础热力图**: 带数值注释的标准热力图
- **相关性矩阵**: 自动计算并可视化相关性
- **聚类热力图**: 带层次聚类的热力图
- **图像式热力图**: 连续数据的可视化

**特点**:
- ✅ 智能颜色映射
- ✅ 自动格式化数值
- ✅ 灵活的标签设置
- ✅ 支持多种聚类方法

**使用示例**:
```python
from visualization.advanced_templates.heatmaps import plot_correlation_matrix
import pandas as pd

df = pd.DataFrame(np.random.rand(100, 4), 
                  columns=['A', 'B', 'C', 'D'])

# 绘制相关性矩阵
fig, ax = plot_correlation_matrix(
    df,
    method='pearson',
    cmap='RdYlBu_r',
    save_path='correlation.png'
)
```

### 3. 专业配色方案 (`professional_colors.py`)

提供经过验证的专业配色方案：

#### 期刊官方配色
- **Nature 配色**: Nature 期刊官方使用的配色方案
- **Science 配色**: Science 期刊官方配色

#### 建模竞赛配色
- **专业风格**: 适合建模竞赛的专业配色
- **优雅风格**: 优雅的配色组合
- **鲜艳风格**: 醒目的配色方案

#### 可访问性配色
- **色盲友好**: 符合 WCAG 2.1 标准的配色
- **高对比度**: 确保清晰可读

**使用示例**:
```python
from visualization.advanced_templates.professional_colors import (
    get_modeling_competition_palette,
    get_journal_palette,
    get_colorblind_safe_palette
)

# 获取建模竞赛配色
colors = get_modeling_competition_palette('professional')

# 获取 Nature 期刊配色
nature_colors = get_journal_palette('nature')

# 获取色盲友好配色
cb_colors = get_colorblind_safe_palette(n_colors=5)

# 使用配色
for i, (x, y) in enumerate(data_list):
    plt.plot(x, y, color=colors[i], label=f'Line {i+1}')
```

## 🎨 配色方案对比

### Nature 配色（推荐用于论文）
```
专业、学术、经典
├── 主色: #E64B35 (红)
├── 辅色: #4DBBD5 (青)
├── 第三: #00A087 (绿)
└── 适合: 学术论文、研究报告
```

### 建模竞赛配色（推荐用于比赛）
```
现代、醒目、专业
├── 风格1: professional (专业商务)
├── 风格2: elegant (优雅学术)
└── 风格3: vibrant (鲜明活泼)
```

### 色盲友好配色（推荐用于演示）
```
可访问、清晰、安全
├── 8种区分度高
├── 符合 WCAG 标准
└── 适合: 演示、海报
```

## 📐 图表规范对比

| 特性 | 标准 | Nature | Science | IEEE | 建模竞赛 |
|------|------|--------|---------|------|----------|
| **单栏尺寸** | - | 3.5" | 3.3" | 3.5" | 4" |
| **双栏尺寸** | - | 7.0" | 6.5" | 7.0" | 8" |
| **最小字号** | - | 6pt | 7pt | 8pt | 10pt |
| **线条宽度** | - | 1.0 | 1.2 | 1.5 | 2.0 |
| **分辨率** | 300 | 600 | 600 | 300-600 | 300 |
| **字体** | Arial | Arial | Arial | Times | Arial |

## 🚀 快速使用指南

### 场景1: 投稿到顶级期刊

```python
from visualization.advanced_templates import plot_publication_figure
from visualization.advanced_templates.professional_colors import get_journal_palette

# 应用 Nature 风格
fig, ax = plot_publication_figure(
    x, y,
    style='nature',
    color=get_journal_palette('nature')[0],
    save_path='nature_figure.png',
    dpi=600
)
```

### 场景2: 数学建模竞赛

```python
from visualization.advanced_templates import plot_publication_figure
from visualization.advanced_templates.professional_colors import get_modeling_competition_palette

# 使用建模竞赛配色
colors = get_modeling_competition_palette('professional')

fig, ax = plot_publication_figure(
    x, y,
    style='nature',  # 使用期刊风格确保专业性
    color=colors[0],
    figsize=(8, 5),  # 稍大以便展示
    save_path='competition_figure.png',
    dpi=300
)
```

### 场景3: 创建复杂多面板图表

```python
from visualization.advanced_templates import plot_multi_panel_figure
from visualization.advanced_templates.professional_colors import NATURE_PALETTE

data_list = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

fig, axes = plot_multi_panel_figure(
    data_list,
    nrows=2, ncols=2,
    style='nature',
    labels=['A', 'B', 'C', 'D'],
    save_path='multi_panel.png',
    dpi=600
)

# 为每个子图使用不同颜色
for i, ax in enumerate(axes.flat):
    ax.plot([], [], color=NATURE_PALETTE[i])
```

## 💡 最佳实践

### 1. 配色选择
- **学术论文**: 使用 Nature 或 Science 配色
- **建模竞赛**: 使用 professional 配色
- **演示报告**: 使用 vibrant 或色盲友好配色
- **黑白打印**: 使用 diverging_colormaps

### 2. 图表尺寸
- **单栏图**: 3.5" (Nature)
- **双栏图**: 7.0" (Nature)
- **幻灯片**: 10" x 7.5"
- **海报**: 根据布局调整

### 3. 分辨率设置
- **期刊投稿**: 600 DPI
- **打印**: 300 DPI
- **屏幕预览**: 100-150 DPI
- **在线发布**: 150-200 DPI

### 4. 字体选择
- **英文**: Arial, Helvetica
- **中文**: SimHei, Microsoft YaHei
- **数学公式**: Times New Roman

## 🔗 相关资源

- **Nature 图表指南**: https://research-figure-guide.nature.com/
- **SciencePlots**: https://github.com/garrettj403/SciencePlots
- **色盲友好配色**: Wong (2011), Nature Methods

## 📚 参考资料

1. Nature Research. "Preparing figures - our specifications"
2. Science Journals. "Information for Authors: Figures"
3. IEEE. "IEEE Graphics Requirements for Accepted Manuscripts"
4. Wong, B. "Points of view: Color coding." Nature Methods (2011)

---

**提示**: 这些模板基于顶级期刊和获奖作品的最佳实践，确保您的图表达到专业水准！
