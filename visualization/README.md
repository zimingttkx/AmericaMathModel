# 可视化指南 📊

> 专业的数学建模图表绘制模板库

## 🎨 美赛图表要求

### 基本要求
1. ✅ **高分辨率**: 至少 300 DPI，建议 600 DPI
2. ✅ **专业配色**: 使用学术配色方案，避免过于鲜艳
3. ✅ **清晰标注**: 标题、坐标轴标签、图例必须清晰
4. ✅ **统一字体**: Times New Roman 或 Arial，字号适中
5. ✅ **适当留白**: 不要过于拥挤，保持视觉舒适

### 图表规范
- **图表编号**: Figure 1, Figure 2, ...
- **标题位置**: 图表下方居中
- **坐标轴**: 必须有单位和标签
- **图例**: 位置合理，不遮挡数据
- **颜色**: 考虑色盲友好，避免红绿搭配

## 📊 图表选择指南

### 按数据类型选择

| 数据类型 | 推荐图表 | 模板位置 |
|---------|---------|---------|
| **时间序列** | 折线图、面积图 | [`matplotlib_templates/line_charts.py`](matplotlib_templates/line_charts.py) |
| **分类数据** | 柱状图、条形图 | [`matplotlib_templates/bar_charts.py`](matplotlib_templates/bar_charts.py) |
| **连续数据** | 散点图、等高线图 | [`matplotlib_templates/scatter_plots.py`](matplotlib_templates/scatter_plots.py) |
| **相关性** | 热力图、散点矩阵 | [`matplotlib_templates/heatmaps.py`](matplotlib_templates/heatmaps.py) |
| **三维数据** | 3D曲面、3D散点 | [`matplotlib_templates/3d_plots.py`](matplotlib_templates/3d_plots.py) |
| **网络关系** | 网络图、桑基图 | [`networkx_templates/`](networkx_templates/) |

### 按展示目的选择

| 展示目的 | 推荐图表 | 说明 |
|---------|---------|------|
| **展示趋势** | 折线图、面积图 | 显示数据随时间或其他变量的变化 |
| **比较数值** | 柱状图、雷达图 | 比较不同类别或组别的数值 |
| **展示分布** | 直方图、箱线图、小提琴图 | 显示数据的分布特征 |
| **展示关系** | 散点图、热力图 | 显示变量之间的相关性 |
| **展示比例** | 饼图、堆叠图 | 显示部分与整体的关系 |
| **展示流程** | 桑基图、流程图 | 显示流量或过程 |

## 🎯 快速使用

### 基础配置

```python
from visualization.style_config import paper_style

# 应用论文风格（全局设置）
paper_style.apply()

# 或者使用上下文管理器（局部设置）
with paper_style.context():
    # 在这里绘制图表
    plt.plot(x, y)
    plt.savefig('figure.png', dpi=300)
```

### 示例1: 绘制折线图

```python
from visualization.matplotlib_templates import line_charts
import numpy as np

# 准备数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制多条折线
line_charts.plot_multi_lines(
    x=x,
    y_list=[y1, y2],
    labels=['sin(x)', 'cos(x)'],
    title='Trigonometric Functions',
    xlabel='x',
    ylabel='y',
    save_path='figures/trig_functions.png',
    dpi=300
)
```

### 示例2: 绘制热力图

```python
from visualization.matplotlib_templates import heatmaps
import numpy as np

# 准备相关性矩阵
correlation_matrix = np.random.rand(10, 10)

# 绘制热力图
heatmaps.plot_correlation_heatmap(
    data=correlation_matrix,
    labels=['Var' + str(i) for i in range(10)],
    title='Correlation Matrix',
    save_path='figures/correlation.png',
    cmap='coolwarm',
    dpi=300
)
```

### 示例3: 绘制3D曲面

```python
from visualization.matplotlib_templates import plots_3d
import numpy as np

# 准备数据
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 绘制3D曲面
plots_3d.plot_surface(
    X=X, Y=Y, Z=Z,
    title='3D Surface Plot',
    xlabel='X', ylabel='Y', zlabel='Z',
    save_path='figures/surface.png',
    dpi=300
)
```

### 示例4: 绘制网络图

```python
from visualization.networkx_templates import graph_visualization
import networkx as nx

# 创建图
G = nx.karate_club_graph()

# 绘制网络图
graph_visualization.plot_network(
    G=G,
    title='Social Network',
    node_size=500,
    layout='spring',
    save_path='figures/network.png',
    dpi=300
)
```

## 🎨 配色方案

### 学术配色

```python
from visualization.style_config import color_schemes

# 使用预定义配色方案
colors = color_schemes.ACADEMIC_COLORS  # 学术蓝色系
colors = color_schemes.NATURE_COLORS    # 自然绿色系
colors = color_schemes.WARM_COLORS      # 暖色系
colors = color_schemes.COOL_COLORS      # 冷色系
colors = color_schemes.COLORBLIND_SAFE  # 色盲友好

# 使用示例
plt.plot(x, y1, color=colors[0], label='Line 1')
plt.plot(x, y2, color=colors[1], label='Line 2')
```

### 自定义配色

```python
# 定义自己的配色方案
my_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# 应用到图表
for i, (x, y) in enumerate(data_list):
    plt.plot(x, y, color=my_colors[i % len(my_colors)])
```

## 📐 图表尺寸建议

### 常用尺寸（英寸）

```python
# 单列图（适合论文单列）
figsize = (6, 4)

# 双列图（适合论文双列）
figsize = (12, 4)

# 方形图（适合对称数据）
figsize = (6, 6)

# 宽屏图（适合时间序列）
figsize = (10, 4)

# 使用示例
plt.figure(figsize=(6, 4))
plt.plot(x, y)
```

### DPI 设置

```python
# 屏幕预览
dpi = 100

# 论文打印
dpi = 300

# 高质量出版
dpi = 600

# 保存图片
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
```

## 🔧 高级技巧

### 1. 子图布局

```python
import matplotlib.pyplot as plt

# 创建2x2子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 在每个子图中绘制
axes[0, 0].plot(x, y1)
axes[0, 0].set_title('Subplot 1')

axes[0, 1].plot(x, y2)
axes[0, 1].set_title('Subplot 2')

# 调整子图间距
plt.tight_layout()
plt.savefig('subplots.png', dpi=300)
```

### 2. 双Y轴图表

```python
fig, ax1 = plt.subplots(figsize=(8, 5))

# 第一个Y轴
ax1.plot(x, y1, 'b-', label='Data 1')
ax1.set_xlabel('X')
ax1.set_ylabel('Y1', color='b')
ax1.tick_params(axis='y', labelcolor='b')

# 第二个Y轴
ax2 = ax1.twinx()
ax2.plot(x, y2, 'r-', label='Data 2')
ax2.set_ylabel('Y2', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Dual Y-axis Plot')
plt.savefig('dual_axis.png', dpi=300, bbox_inches='tight')
```

### 3. 添加注释

```python
plt.plot(x, y)

# 添加文本注释
plt.text(5, 10, 'Important Point', fontsize=12)

# 添加箭头注释
plt.annotate('Peak', xy=(5, 10), xytext=(6, 12),
             arrowprops=dict(arrowstyle='->', color='red'))

# 添加水平/垂直线
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=5, color='k', linestyle='--', alpha=0.3)
```

### 4. 图例优化

```python
plt.plot(x, y1, label='Method 1')
plt.plot(x, y2, label='Method 2')

# 自定义图例位置
plt.legend(loc='upper right')  # 右上角
plt.legend(loc='best')         # 自动选择最佳位置
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图外

# 多列图例
plt.legend(ncol=2)

# 无边框图例
plt.legend(frameon=False)
```

## 📋 模板库结构

```
visualization/
├── matplotlib_templates/      # Matplotlib 基础模板
│   ├── line_charts.py         # 折线图
│   ├── bar_charts.py          # 柱状图
│   ├── scatter_plots.py       # 散点图
│   ├── heatmaps.py            # 热力图
│   └── 3d_plots.py            # 3D图形
├── seaborn_templates/         # Seaborn 高级模板
│   ├── distribution_plots.py  # 分布图
│   ├── regression_plots.py    # 回归图
│   └── categorical_plots.py   # 分类图
├── plotly_templates/          # Plotly 交互式图表
│   ├── interactive_charts.py  # 交互式图表
│   └── animations.py          # 动画图表
├── networkx_templates/        # 网络图可视化
│   └── graph_visualization.py # 图论可视化
└── style_config/              # 样式配置
    ├── color_schemes.py       # 配色方案
    ├── fonts.py               # 字体配置
    └── paper_style.py         # 论文风格
```

## ⚠️ 常见问题

### 1. 中文显示问题

```python
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
```

### 2. 图片保存问题

```python
# 保存时裁剪空白
plt.savefig('figure.png', bbox_inches='tight')

# 保存为矢量图（适合放大）
plt.savefig('figure.pdf')
plt.savefig('figure.svg')

# 保存为透明背景
plt.savefig('figure.png', transparent=True)
```

### 3. 内存问题

```python
# 绘制完成后关闭图形
plt.close()

# 或使用上下文管理器
with plt.style.context('seaborn'):
    plt.plot(x, y)
    plt.savefig('figure.png')
# 自动关闭
```

## 🔗 相关资源

- **算法库**: [`../algorithms/`](../algorithms/)
- **完整示例**: [`../examples/`](../examples/)
- **数据处理**: [`../utils/data_processing.py`](../utils/data_processing.py)

## 📚 推荐学习资源

- [Matplotlib 官方文档](https://matplotlib.org/)
- [Seaborn 官方文档](https://seaborn.pydata.org/)
- [Plotly 官方文档](https://plotly.com/python/)
- [NetworkX 官方文档](https://networkx.org/)

---

**提示**: 所有模板都支持自定义参数，详见各模板文件的文档字符串！
