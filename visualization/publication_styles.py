"""
顶会论文级可视化样式模块
Publication-Quality Visualization Styles

参考来源:
- Nature/Science 期刊图表规范
- IEEE 论文图表标准
- NeurIPS/ICML 顶会风格
- SciencePlots 库最佳实践

使用方法:
    from visualization.publication_styles import apply_nature_style, apply_ieee_style
    apply_nature_style()  # 应用 Nature 风格
    plt.plot(x, y)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cycler import cycler

# ============================================================================
# 顶会/期刊配色方案 (Color Palettes)
# ============================================================================

# Nature 期刊配色 - 清晰、专业
NATURE_COLORS = [
    '#4C72B0',  # 蓝色
    '#DD8452',  # 橙色
    '#55A868',  # 绿色
    '#C44E52',  # 红色
    '#8172B3',  # 紫色
    '#937860',  # 棕色
    '#DA8BC3',  # 粉色
    '#8C8C8C',  # 灰色
    '#CCB974',  # 黄色
    '#64B5CD',  # 青色
]

# IEEE 配色 - 黑白友好，适合打印
IEEE_COLORS = [
    '#000000',  # 黑色
    '#E69F00',  # 橙色
    '#56B4E9',  # 天蓝
    '#009E73',  # 绿色
    '#F0E442',  # 黄色
    '#0072B2',  # 深蓝
    '#D55E00',  # 深橙
    '#CC79A7',  # 粉紫
]

# Science 期刊配色
SCIENCE_COLORS = [
    '#0C5DA5',  # 深蓝
    '#00B945',  # 绿色
    '#FF9500',  # 橙色
    '#FF2C00',  # 红色
    '#845B97',  # 紫色
    '#474747',  # 深灰
    '#9E9E9E',  # 浅灰
]

# 色盲友好配色 (Paul Tol's Bright)
COLORBLIND_SAFE = [
    '#4477AA',  # 蓝色
    '#EE6677',  # 红色
    '#228833',  # 绿色
    '#CCBB44',  # 黄色
    '#66CCEE',  # 青色
    '#AA3377',  # 紫色
    '#BBBBBB',  # 灰色
]

# 高对比度配色 (适合演示)
HIGH_CONTRAST = [
    '#E74C3C',  # 红色
    '#3498DB',  # 蓝色
    '#2ECC71',  # 绿色
    '#F39C12',  # 橙色
    '#9B59B6',  # 紫色
    '#1ABC9C',  # 青色
    '#34495E',  # 深灰
]

# ============================================================================
# 期刊尺寸规范 (Figure Dimensions)
# ============================================================================

# Nature 尺寸 (mm -> inches, 1 inch = 25.4 mm)
NATURE_SINGLE_COL = 89 / 25.4   # 3.5 inches
NATURE_1_5_COL = 120 / 25.4     # 4.7 inches
NATURE_DOUBLE_COL = 183 / 25.4  # 7.2 inches

# IEEE 尺寸
IEEE_SINGLE_COL = 3.5   # inches
IEEE_DOUBLE_COL = 7.16  # inches

# Science 尺寸
SCIENCE_SINGLE_COL = 2.3  # inches
SCIENCE_DOUBLE_COL = 4.6  # inches

# ============================================================================
# 样式配置函数
# ============================================================================

def apply_nature_style():
    """
    应用 Nature 期刊风格
    - Sans-serif 字体 (Helvetica/Arial)
    - 最小字号 5pt，推荐 7pt
    - RGB 颜色模式
    """
    plt.rcParams.update({
        # 字体设置
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        
        # 坐标轴
        'axes.labelsize': 7,
        'axes.titlesize': 8,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': cycler('color', NATURE_COLORS),
        
        # 刻度
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        
        # 图例
        'legend.fontsize': 6,
        'legend.frameon': False,
        'legend.borderpad': 0.4,
        
        # 线条
        'lines.linewidth': 1.0,
        'lines.markersize': 4,
        
        # 图形
        'figure.figsize': (NATURE_SINGLE_COL, NATURE_SINGLE_COL * 0.75),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        
        # 网格
        'axes.grid': False,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.3,
    })
    print("Applied Nature journal style")


def apply_ieee_style():
    """
    应用 IEEE 论文风格
    - 黑白打印友好
    - Times New Roman 字体
    - 单栏宽度 3.5 inches
    """
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        
        # 坐标轴
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.linewidth': 0.6,
        'axes.prop_cycle': cycler('color', IEEE_COLORS),
        
        # 刻度
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        
        # 图例
        'legend.fontsize': 7,
        'legend.frameon': True,
        'legend.edgecolor': 'black',
        'legend.fancybox': False,
        
        # 线条
        'lines.linewidth': 1.0,
        'lines.markersize': 5,
        
        # 图形
        'figure.figsize': (IEEE_SINGLE_COL, IEEE_SINGLE_COL * 0.75),
        'figure.dpi': 300,
        'savefig.dpi': 600,  # IEEE 要求高分辨率
        'savefig.bbox': 'tight',
        
        # 网格
        'axes.grid': True,
        'grid.linewidth': 0.3,
        'grid.linestyle': '--',
        'grid.alpha': 0.5,
    })
    print("Applied IEEE paper style")


def apply_science_style():
    """
    应用 Science 期刊风格
    - 简洁、现代
    - Sans-serif 字体
    """
    plt.rcParams.update({
        # 字体设置
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 8,
        
        # 坐标轴
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': cycler('color', SCIENCE_COLORS),
        
        # 刻度
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        
        # 图例
        'legend.fontsize': 7,
        'legend.frameon': False,
        
        # 线条
        'lines.linewidth': 1.2,
        'lines.markersize': 5,
        
        # 图形
        'figure.figsize': (SCIENCE_DOUBLE_COL, SCIENCE_DOUBLE_COL * 0.6),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        # 网格
        'axes.grid': False,
    })
    print("Applied Science journal style")


def apply_neurips_style():
    """
    应用 NeurIPS/ICML 顶会风格
    - 现代、清晰
    - 适合机器学习论文
    """
    plt.rcParams.update({
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times'],
        'font.size': 10,
        
        # 坐标轴
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'axes.linewidth': 0.8,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.prop_cycle': cycler('color', COLORBLIND_SAFE),
        
        # 刻度
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        
        # 图例
        'legend.fontsize': 9,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': '0.8',
        
        # 线条
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # 图形
        'figure.figsize': (5.5, 4.0),
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        
        # 网格
        'axes.grid': True,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    print("Applied NeurIPS/ICML style")


def apply_presentation_style():
    """
    应用演示/PPT风格
    - 大字体、高对比度
    - 适合投影展示
    """
    plt.rcParams.update({
        # 字体设置
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 14,
        
        # 坐标轴
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': cycler('color', HIGH_CONTRAST),
        
        # 刻度
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        
        # 图例
        'legend.fontsize': 12,
        'legend.frameon': False,
        
        # 线条
        'lines.linewidth': 2.5,
        'lines.markersize': 10,
        
        # 图形
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        
        # 网格
        'axes.grid': True,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.3,
    })
    print("Applied presentation style")


def reset_style():
    """重置为 matplotlib 默认样式"""
    plt.rcdefaults()
    print("Reset to matplotlib defaults")


# ============================================================================
# 专业图表绘制函数
# ============================================================================

def create_figure(style='nature', nrows=1, ncols=1, width='single', 
                  height_ratio=0.75, **kwargs):
    """
    创建符合期刊规范的图形
    
    Parameters:
    -----------
    style : str
        'nature', 'ieee', 'science', 'neurips', 'presentation'
    nrows, ncols : int
        子图行列数
    width : str or float
        'single', 'double', '1.5' 或具体英寸数
    height_ratio : float
        高度与宽度的比例
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    # 应用样式
    style_funcs = {
        'nature': apply_nature_style,
        'ieee': apply_ieee_style,
        'science': apply_science_style,
        'neurips': apply_neurips_style,
        'presentation': apply_presentation_style,
    }
    if style in style_funcs:
        style_funcs[style]()
    
    # 确定宽度
    width_map = {
        'nature': {'single': NATURE_SINGLE_COL, 'double': NATURE_DOUBLE_COL, '1.5': NATURE_1_5_COL},
        'ieee': {'single': IEEE_SINGLE_COL, 'double': IEEE_DOUBLE_COL, '1.5': 5.0},
        'science': {'single': SCIENCE_SINGLE_COL, 'double': SCIENCE_DOUBLE_COL, '1.5': 3.5},
    }
    
    if isinstance(width, str) and style in width_map:
        fig_width = width_map[style].get(width, width_map[style]['single'])
    elif isinstance(width, (int, float)):
        fig_width = width
    else:
        fig_width = 5.0
    
    fig_height = fig_width * height_ratio * nrows / ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), **kwargs)
    
    return fig, axes


def add_panel_labels(axes, labels=None, fontsize=10, fontweight='bold', 
                     loc='upper left', offset=(-0.1, 1.05)):
    """
    为多面板图添加标签 (a), (b), (c)...
    
    Parameters:
    -----------
    axes : array of axes
        子图数组
    labels : list, optional
        自定义标签，默认为 a, b, c...
    fontsize : int
        字体大小
    fontweight : str
        字体粗细
    loc : str
        位置
    offset : tuple
        偏移量 (x, y)
    """
    if not hasattr(axes, '__iter__'):
        axes = [axes]
    else:
        axes = axes.flatten()
    
    if labels is None:
        labels = [chr(ord('a') + i) for i in range(len(axes))]
    
    for ax, label in zip(axes, labels):
        ax.text(offset[0], offset[1], f'({label})', transform=ax.transAxes,
                fontsize=fontsize, fontweight=fontweight, va='top', ha='left')


def save_publication_figure(fig, filename, formats=['pdf', 'png'], dpi=300):
    """
    保存为多种格式的出版级图形
    
    Parameters:
    -----------
    fig : matplotlib figure
    filename : str
        文件名（不含扩展名）
    formats : list
        输出格式列表
    dpi : int
        分辨率
    """
    for fmt in formats:
        fig.savefig(f'{filename}.{fmt}', format=fmt, dpi=dpi, 
                    bbox_inches='tight', pad_inches=0.02)
        print(f"Saved: {filename}.{fmt}")


# ============================================================================
# 专业图表类型
# ============================================================================

def plot_with_error_band(ax, x, y, yerr, color=None, label=None, alpha=0.2):
    """
    绘制带误差带的曲线（常用于机器学习论文）
    
    Parameters:
    -----------
    ax : matplotlib axes
    x : array-like
    y : array-like
        均值
    yerr : array-like
        标准差或误差
    """
    if color is None:
        color = NATURE_COLORS[0]
    
    line, = ax.plot(x, y, color=color, label=label, linewidth=1.5)
    ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=alpha)
    
    return line


def plot_comparison_bars(ax, data, labels, group_labels, colors=None, 
                         width=0.8, show_values=True):
    """
    绘制分组柱状图（算法对比常用）
    
    Parameters:
    -----------
    ax : matplotlib axes
    data : 2D array-like
        shape: (n_groups, n_bars)
    labels : list
        每组的标签
    group_labels : list
        每个柱子的标签
    """
    if colors is None:
        colors = NATURE_COLORS[:len(group_labels)]
    
    n_groups = len(labels)
    n_bars = len(group_labels)
    bar_width = width / n_bars
    x = np.arange(n_groups)
    
    for i, (group_label, color) in enumerate(zip(group_labels, colors)):
        offset = (i - n_bars/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, data[:, i], bar_width, label=group_label, 
                      color=color, edgecolor='black', linewidth=0.5)
        
        if show_values:
            for bar, val in zip(bars, data[:, i]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.2f}', ha='center', va='bottom', fontsize=6)
    
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    
    return ax


def plot_heatmap_publication(ax, data, row_labels, col_labels, cmap='RdYlBu_r',
                             annotate=True, fmt='.2f', cbar_label=None):
    """
    绘制出版级热力图
    
    Parameters:
    -----------
    ax : matplotlib axes
    data : 2D array-like
    row_labels, col_labels : list
        行列标签
    cmap : str
        颜色映射
    annotate : bool
        是否显示数值
    """
    im = ax.imshow(data, cmap=cmap, aspect='auto')
    
    # 设置刻度
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    # 旋转 x 轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # 添加数值标注
    if annotate:
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = data[i, j]
                # 根据背景色选择文字颜色
                text_color = 'white' if val > (data.max() + data.min()) / 2 else 'black'
                ax.text(j, i, format(val, fmt), ha='center', va='center', 
                        color=text_color, fontsize=7)
    
    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    
    return im, cbar


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == '__main__':
    # 示例：创建 Nature 风格图表
    fig, ax = create_figure(style='nature', width='single')
    
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    ax.plot(x, y1, label='sin(x)')
    ax.plot(x, y2, label='cos(x)')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend()
    
    save_publication_figure(fig, 'example_nature', formats=['png'])
    plt.show()
