"""
论文风格配置模块
提供符合学术论文要求的图表样式设置
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from contextlib import contextmanager


class PaperStyle:
    """论文风格配置类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        'figure.figsize': (6, 4),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # 字体设置
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # 线条设置
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'axes.linewidth': 1,
        
        # 网格设置
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        
        # 图例设置
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'gray',
        
        # 坐标轴设置
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        
        # 颜色设置
        'axes.prop_cycle': plt.cycler('color', [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
            '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'
        ])
    }
    
    @classmethod
    def apply(cls, config=None):
        """
        应用论文风格（全局设置）
        
        Parameters:
        -----------
        config : dict, optional
            自定义配置，会覆盖默认配置
        """
        style_config = cls.DEFAULT_CONFIG.copy()
        if config:
            style_config.update(config)
        
        for key, value in style_config.items():
            mpl.rcParams[key] = value
    
    @classmethod
    @contextmanager
    def context(cls, config=None):
        """
        使用上下文管理器临时应用论文风格
        
        Parameters:
        -----------
        config : dict, optional
            自定义配置
            
        Example:
        --------
        >>> with PaperStyle.context():
        ...     plt.plot(x, y)
        ...     plt.savefig('figure.png')
        """
        style_config = cls.DEFAULT_CONFIG.copy()
        if config:
            style_config.update(config)
        
        with plt.style.context(style_config):
            yield
    
    @staticmethod
    def set_chinese_font():
        """设置中文字体支持"""
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    @staticmethod
    def save_figure(fig, filepath, dpi=300, **kwargs):
        """
        保存图表（推荐使用此方法）
        
        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            图表对象
        filepath : str
            保存路径
        dpi : int
            分辨率
        **kwargs : dict
            其他参数传递给 savefig
        """
        default_kwargs = {
            'dpi': dpi,
            'bbox_inches': 'tight',
            'pad_inches': 0.1,
            'facecolor': 'white',
            'edgecolor': 'none'
        }
        default_kwargs.update(kwargs)
        fig.savefig(filepath, **default_kwargs)
        print(f"图表已保存至: {filepath}")


# 预定义样式
class Styles:
    """预定义的图表样式"""
    
    MINIMAL = {
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.grid': False,
    }
    
    GRID = {
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
    }
    
    PRESENTATION = {
        'font.size': 14,
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'lines.linewidth': 2,
    }
    
    POSTER = {
        'font.size': 18,
        'axes.labelsize': 22,
        'axes.titlesize': 26,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'lines.linewidth': 3,
    }


# 创建全局实例
paper_style = PaperStyle()


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    
    # 应用论文风格
    paper_style.apply()
    
    # 生成测试数据
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y1, label='sin(x)')
    ax.plot(x, y2, label='cos(x)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Paper Style Example')
    ax.legend()
    
    # 保存图表
    paper_style.save_figure(fig, 'test_paper_style.png')
    plt.show()
