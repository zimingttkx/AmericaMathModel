"""
高级图表模板模块
提供顶级期刊和建模竞赛级别的图表模板
"""

from .heatmaps import *
from .scientific_plots import *
from .professional_colors import *

__all__ = [
    # 热力图相关
    'plot_heatmap',
    'plot_clustermap',
    'plot_correlation_matrix',
    'plot_imshow',
    # 科学出版物级图表
    'plot_publication_figure',
    'plot_multi_panel_figure',
    'plot_with_annotations',
    'PublicationStyle',
    # 专业配色
    'get_color_scheme',
    'create_custom_colormap',
]
