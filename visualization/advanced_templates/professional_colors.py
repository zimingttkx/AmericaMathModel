"""
专业配色方案
基于顶级期刊和建模竞赛获奖作品的配色
"""

# Nature 期刊官方配色
NATURE_COLORS = {
    'primary': '#E64B35',      # 红色
    'secondary': '#4DBBD5',    # 青色
    'tertiary': '#00A087',     # 绿色
    'quaternary': '#3C5488',   # 蓝色
    'quinquennial': '#F39B7F', # 橙色
    'senary': '#8491B4',       # 紫色
}

NATURE_PALETTE = [
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488',
    '#F39B7F', '#8491B4', '#91D1C2', '#DC0000',
    '#7E6148', '#B09C85'
]

# Science 期刊配色
SCIENCE_COLORS = {
    'blue': '#3B4992',
    'red': '#EE0000',
    'green': '#008B45',
    'purple': '#631879',
    'cyan': '#008280',
    'darkred': '#BB0021',
}

SCIENCE_PALETTE = [
    '#3B4992', '#EE0000', '#008B45', '#631879',
    '#008280', '#BB0021', '#79BE9B', '#9A9A32',
]

# 色盲友好配色（Wong, 2011, Nature Methods）
COLORBLIND_SAFE_PALETTE = [
    '#E69F00',  # 橙色
    '#56B4E9',  # 天蓝色
    '#009E73',  # 青绿色
    '#F0E442',  # 黄色
    '#0072B2',  # 蓝色
    '#D55E00',  # 朱红色
    '#CC79A7',  # 红紫色
    '#999999',  # 灰色
]

# MCM/ICM 获奖作品常用配色
MODELING_COMPETITION_COLORS = {
    'professional': [
        '#2C3E50',  # 深蓝灰
        '#E74C3C',  # 红色
        '#3498DB',  # 蓝色
        '#2ECC71',  # 绿色
        '#F39C12',  # 橙色
        '#9B59B6',  # 紫色
    ],
    'elegant': [
        '#34495E',  # 深灰蓝
        '#16A085',  # 青绿色
        '#27AE60',  # 绿色
        '#2980B9',  # 蓝色
        '#8E44AD',  # 紫色
        '#2C3E50',  # 深蓝
    ],
    'vibrant': [
        '#FF6B6B',  # 亮红
        '#4ECDC4',  # 青色
        '#45B7D1',  # 蓝色
        '#FFA07A',  # 浅橙
        '#98D8C8',  # 薄荷绿
        '#F7DC6F',  # 黄色
    ],
}

# 连续配色（用于热力图）
SEQUENTIAL_COLORMAPS = {
    'viridis': 'viridis',      # 紫-蓝-绿-黄（感知均匀）
    'plasma': 'plasma',        # 紫-红-黄
    'inferno': 'inferno',      # 黑-紫-红-黄
    'magma': 'magma',          # 黑-紫-粉-黄
    'cividis': 'cividis',      # 色盲友好
    'rocket': 'rocket',        # 深蓝-红-黄
    'mako': 'mako',            # 深蓝-青-白
}

# 发散配色（用于正负值）
DIVERGING_COLORMAPS = {
    'coolwarm': 'coolwarm',           # 蓝-白-红
    'RdBu': 'RdBu_r',                 # 红-白-蓝
    'RdYlBu': 'RdYlBu_r',             # 红-黄-蓝
    'RdYlGn': 'RdYlGn_r',             # 红-黄-绿
    'PiYG': 'PiYG',                   # 粉-白-绿
    'BrBG': 'BrBG',                   # 棕-白-绿
    'seismic': 'seismic',             # 蓝-白-红
}

# 定性配色（用于分类数据）
QUALITATIVE_COLORMAPS = {
    'Set1': 'Set1',
    'Set2': 'Set2',
    'Set3': 'Set3',
    'tab10': 'tab10',
    'tab20': 'tab20',
    'Paired': 'Paired',
    'Accent': 'Accent',
}


def get_modeling_competition_palette(style: str = 'professional') -> list:
    """
    获取建模竞赛配色方案
    
    Parameters:
    -----------
    style : str
        配色风格: 'professional', 'elegant', 'vibrant'
    
    Returns:
    --------
    colors : list
        颜色列表
    
    Example:
    --------
    >>> colors = get_modeling_competition_palette('professional')
    >>> print(colors)
    """
    return MODELING_COMPETITION_COLORS.get(style, MODELING_COMPETITION_COLORS['professional'])


def get_journal_palette(journal: str = 'nature') -> list:
    """
    获取期刊配色方案
    
    Parameters:
    -----------
    journal : str
        期刊名称: 'nature', 'science'
    
    Returns:
    --------
    colors : list
        颜色列表
    
    Example:
    --------
    >>> colors = get_journal_palette('nature')
    """
    if journal.lower() == 'nature':
        return NATURE_PALETTE
    elif journal.lower() == 'science':
        return SCIENCE_PALETTE
    else:
        return NATURE_PALETTE


def get_colorblind_safe_palette(n_colors: int = 8) -> list:
    """
    获取色盲友好配色
    
    Parameters:
    -----------
    n_colors : int
        需要的颜色数量
    
    Returns:
    --------
    colors : list
        颜色列表
    """
    return COLORBLIND_SAFE_PALETTE[:min(n_colors, len(COLORBLIND_SAFE_PALETTE))]


def get_gradient_palette(start_color: str, end_color: str, n_colors: int = 10) -> list:
    """
    生成渐变配色
    
    Parameters:
    -----------
    start_color : str
        起始颜色（hex）
    end_color : str
        结束颜色（hex）
    n_colors : int
        颜色数量
    
    Returns:
    --------
    colors : list
        渐变颜色列表
    
    Example:
    --------
    >>> colors = get_gradient_palette('#1f77b4', '#d62728', n_colors=5)
    """
    import matplotlib.colors as mcolors
    import numpy as np
    
    # 将 hex 转换为 RGB
    start_rgb = mcolors.hex2color(start_color)
    end_rgb = mcolors.hex2color(end_color)
    
    # 生成渐变
    colors = []
    for i in range(n_colors):
        ratio = i / (n_colors - 1) if n_colors > 1 else 0
        r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
        g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
        b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
        colors.append(mcolors.to_hex((r, g, b)))
    
    return colors


def get_contrasting_color(hex_color: str) -> str:
    """
    获取对比色（黑色或白色）
    
    Parameters:
    -----------
    hex_color : str
        输入颜色（hex）
    
    Returns:
    --------
    contrast_color : str
        对比色（'#000000' 或 '#ffffff'）
    """
    import matplotlib.colors as mcolors
    
    rgb = mcolors.hex2color(hex_color)
    # 计算亮度
    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
    
    if luminance > 0.5:
        return '#000000'  # 黑色
    else:
        return '#ffffff'  # 白色


# 预定义的图表配色组合
CHART_COLOR_COMBINATIONS = {
    'bar_chart_pairs': [
        ('#1f77b4', '#aec7e8'),  # 蓝色对
        ('#ff7f0e', '#ffbb78'),  # 橙色对
        ('#2ca02c', '#98df8a'),  # 绿色对
        ('#d62728', '#ff9896'),  # 红色对
    ],
    'line_comparison': [
        '#1f77b4',  # 主要数据
        '#ff7f0e',  # 次要数据
        '#2ca02c',  # 第三数据
        '#d62728',  # 第四数据
    ],
    'heatmap_sequential': [
        '#f7fbff', '#deebf7', '#c6dbef', '#9ecae1',
        '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'
    ],
    'heatmap_diverging': [
        '#67001f', '#b2182b', '#d6604d', '#f4a582',
        '#fddbc7', '#f7f7f7', '#d1e5f0', '#92c5de',
        '#4393c3', '#2166ac', '#053061'
    ],
}


if __name__ == '__main__':
    # 测试代码
    print("测试专业配色方案...")
    
    # 测试1: 建模竞赛配色
    print("\n建模竞赛配色:")
    for style, colors in MODELING_COMPETITION_COLORS.items():
        print(f"  {style}: {colors}")
    
    # 测试2: 期刊配色
    print(f"\nNature 配色: {NATURE_PALETTE}")
    print(f"Science 配色: {SCIENCE_PALETTE}")
    
    # 测试3: 色盲友好配色
    print(f"\n色盲友好配色: {COLORBLIND_SAFE_PALETTE}")
    
    # 测试4: 渐变配色
    gradient = get_gradient_palette('#1f77b4', '#d62728', n_colors=5)
    print(f"\n渐变配色: {gradient}")
    
    # 测试5: 对比色
    contrast = get_contrasting_color('#1f77b4')
    print(f"\n蓝色对比色: {contrast}")
    
    print("\n所有测试完成！")
