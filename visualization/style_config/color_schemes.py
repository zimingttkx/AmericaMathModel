"""
配色方案模块
提供多种学术和专业配色方案
"""

# 学术蓝色系（推荐用于论文）
ACADEMIC_COLORS = [
    '#1f77b4',  # 蓝色
    '#ff7f0e',  # 橙色
    '#2ca02c',  # 绿色
    '#d62728',  # 红色
    '#9467bd',  # 紫色
    '#8c564b',  # 棕色
    '#e377c2',  # 粉色
    '#7f7f7f',  # 灰色
]

# 自然绿色系
NATURE_COLORS = [
    '#2ca02c',  # 绿色
    '#98df8a',  # 浅绿
    '#d62728',  # 红色
    '#ff9896',  # 浅红
    '#1f77b4',  # 蓝色
    '#aec7e8',  # 浅蓝
]

# 暖色系
WARM_COLORS = [
    '#d62728',  # 红色
    '#ff7f0e',  # 橙色
    '#ffbb78',  # 浅橙
    '#e377c2',  # 粉色
    '#f7b6d2',  # 浅粉
]

# 冷色系
COOL_COLORS = [
    '#1f77b4',  # 蓝色
    '#aec7e8',  # 浅蓝
    '#2ca02c',  # 绿色
    '#98df8a',  # 浅绿
    '#9467bd',  # 紫色
    '#c5b0d5',  # 浅紫
]

# 色盲友好配色（推荐）
COLORBLIND_SAFE = [
    '#0173B2',  # 蓝色
    '#DE8F05',  # 橙色
    '#029E73',  # 青绿色
    '#CC78BC',  # 紫色
    '#CA9161',  # 棕色
    '#949494',  # 灰色
    '#ECE133',  # 黄色
    '#56B4E9',  # 天蓝色
]

# 单色渐变（蓝色）
BLUE_GRADIENT = [
    '#08519c',  # 深蓝
    '#3182bd',  # 中蓝
    '#6baed6',  # 浅蓝
    '#9ecae1',  # 很浅蓝
    '#c6dbef',  # 极浅蓝
]

# 单色渐变（红色）
RED_GRADIENT = [
    '#a50f15',  # 深红
    '#de2d26',  # 中红
    '#fb6a4a',  # 浅红
    '#fc9272',  # 很浅红
    '#fcbba1',  # 极浅红
]

# 单色渐变（绿色）
GREEN_GRADIENT = [
    '#006d2c',  # 深绿
    '#31a354',  # 中绿
    '#74c476',  # 浅绿
    '#a1d99b',  # 很浅绿
    '#c7e9c0',  # 极浅绿
]

# 发散配色（红-蓝）
DIVERGING_RED_BLUE = [
    '#b2182b',  # 深红
    '#ef8a62',  # 浅红
    '#fddbc7',  # 极浅红
    '#f7f7f7',  # 白色
    '#d1e5f0',  # 极浅蓝
    '#67a9cf',  # 浅蓝
    '#2166ac',  # 深蓝
]

# 科学出版物配色
SCIENTIFIC = [
    '#E64B35',  # 红色
    '#4DBBD5',  # 青色
    '#00A087',  # 绿色
    '#3C5488',  # 深蓝
    '#F39B7F',  # 橙色
    '#8491B4',  # 紫色
    '#91D1C2',  # 薄荷绿
    '#DC0000',  # 深红
]

# Nature 期刊风格
NATURE_JOURNAL = [
    '#E64B35',  # 红色
    '#4DBBD5',  # 青色
    '#00A087',  # 绿色
    '#3C5488',  # 蓝色
    '#F39B7F',  # 橙色
    '#8491B4',  # 紫色
]

# Science 期刊风格
SCIENCE_JOURNAL = [
    '#3B4992',  # 深蓝
    '#EE0000',  # 红色
    '#008B45',  # 绿色
    '#631879',  # 紫色
    '#008280',  # 青色
    '#BB0021',  # 深红
]

# 热力图配色方案
HEATMAP_COLORS = {
    'viridis': 'viridis',      # 紫-绿-黄
    'plasma': 'plasma',        # 紫-粉-黄
    'inferno': 'inferno',      # 黑-红-黄
    'magma': 'magma',          # 黑-紫-白
    'coolwarm': 'coolwarm',    # 蓝-白-红
    'RdYlBu': 'RdYlBu_r',      # 红-黄-蓝
    'RdYlGn': 'RdYlGn_r',      # 红-黄-绿
    'Spectral': 'Spectral_r',  # 光谱色
}


def get_color_palette(name='academic', n_colors=None):
    """
    获取配色方案
    
    Parameters:
    -----------
    name : str
        配色方案名称
        可选: 'academic', 'nature', 'warm', 'cool', 'colorblind',
              'scientific', 'nature_journal', 'science_journal'
    n_colors : int, optional
        需要的颜色数量，如果超过方案中的颜色数，会循环使用
    
    Returns:
    --------
    list : 颜色列表
    
    Example:
    --------
    >>> colors = get_color_palette('academic', n_colors=5)
    >>> plt.plot(x, y, color=colors[0])
    """
    palettes = {
        'academic': ACADEMIC_COLORS,
        'nature': NATURE_COLORS,
        'warm': WARM_COLORS,
        'cool': COOL_COLORS,
        'colorblind': COLORBLIND_SAFE,
        'scientific': SCIENTIFIC,
        'nature_journal': NATURE_JOURNAL,
        'science_journal': SCIENCE_JOURNAL,
        'blue_gradient': BLUE_GRADIENT,
        'red_gradient': RED_GRADIENT,
        'green_gradient': GREEN_GRADIENT,
    }
    
    if name not in palettes:
        raise ValueError(f"Unknown palette: {name}. Available: {list(palettes.keys())}")
    
    colors = palettes[name]
    
    if n_colors is None:
        return colors
    
    # 如果需要更多颜色，循环使用
    if n_colors > len(colors):
        return [colors[i % len(colors)] for i in range(n_colors)]
    
    return colors[:n_colors]


def show_palette(name='academic'):
    """
    显示配色方案
    
    Parameters:
    -----------
    name : str
        配色方案名称
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    colors = get_color_palette(name)
    n = len(colors)
    
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
        ax.text(i + 0.5, 0.5, color, ha='center', va='center',
                fontsize=8, color='white' if i % 2 == 0 else 'black')
    
    ax.set_title(f'Color Palette: {name}', fontsize=14, pad=10)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_all_palettes():
    """显示所有配色方案"""
    import matplotlib.pyplot as plt
    
    palettes = [
        'academic', 'nature', 'warm', 'cool', 'colorblind',
        'scientific', 'nature_journal', 'science_journal',
        'blue_gradient', 'red_gradient', 'green_gradient'
    ]
    
    fig, axes = plt.subplots(len(palettes), 1, figsize=(12, len(palettes) * 0.8))
    
    for ax, name in zip(axes, palettes):
        colors = get_color_palette(name)
        n = len(colors)
        
        ax.set_xlim(0, n)
        ax.set_ylim(0, 1)
        
        for i, color in enumerate(colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, facecolor=color))
        
        ax.set_title(name, loc='left', fontsize=10)
        ax.axis('off')
    
    plt.suptitle('All Color Palettes', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # 显示所有配色方案
    show_all_palettes()
