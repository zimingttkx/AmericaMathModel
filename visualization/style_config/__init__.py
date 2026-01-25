"""
样式配置包
提供论文级图表样式配置
"""

from .paper_style import paper_style, PaperStyle, Styles
from .color_schemes import (
    get_color_palette,
    show_palette,
    show_all_palettes,
    ACADEMIC_COLORS,
    COLORBLIND_SAFE,
    SCIENTIFIC,
)
from .fonts import font_config, FontConfig

__all__ = [
    'paper_style',
    'PaperStyle',
    'Styles',
    'get_color_palette',
    'show_palette',
    'show_all_palettes',
    'ACADEMIC_COLORS',
    'COLORBLIND_SAFE',
    'SCIENTIFIC',
    'font_config',
    'FontConfig',
]
