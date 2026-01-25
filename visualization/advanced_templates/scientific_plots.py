"""
科学出版物级别图表
基于 Nature、Science 等顶级期刊的图表规范
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Union
import seaborn as sns


class PublicationStyle:
    """
    科学出版物风格配置
    符合 Nature、Science 等顶级期刊要求
    """
    
    # Nature 期刊风格
    NATURE_STYLE = {
        'figure.figsize': (3.5, 2.5),  # 单栏图尺寸
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 6,
        'axes.labelsize': 7,
        'axes.titlesize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'patch.linewidth': 0.5,
    }
    
    # Science 期刊风格
    SCIENCE_STYLE = {
        'figure.figsize': (3.3, 2.5),
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica'],
        'font.size': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.7,
    }
    
    # IEEE 期刊风格
    IEEE_STYLE = {
        'figure.figsize': (3.5, 2.6),
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 8,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'lines.linewidth': 1.5,
    }
    
    # 双栏图尺寸（用于横跨两栏的大图）
    TWO_COLUMN_FIGURE = {
        'figure.figsize': (7.0, 2.5),
    }
    
    # 方形图
    SQUARE_FIGURE = {
        'figure.figsize': (3.5, 3.5),
    }
    
    @classmethod
    def apply_nature_style(cls):
        """应用 Nature 期刊风格"""
        plt.rcParams.update(cls.NATURE_STYLE)
    
    @classmethod
    def apply_science_style(cls):
        """应用 Science 期刊风格"""
        plt.rcParams.update(cls.SCIENCE_STYLE)
    
    @classmethod
    def apply_ieee_style(cls):
        """应用 IEEE 期刊风格"""
        plt.rcParams.update(cls.IEEE_STYLE)
    
    @classmethod
    def apply_custom_style(cls, style_dict: dict):
        """应用自定义风格"""
        plt.rcParams.update(style_dict)


def plot_publication_figure(x: Union[np.ndarray, list],
                            y: Union[np.ndarray, list],
                            figsize: Optional[Tuple[float, float]] = None,
                            xlabel: str = 'X',
                            ylabel: str = 'Y',
                            title: Optional[str] = None,
                            style: str = 'nature',
                            color: str = '#1f77b4',
                            linewidth: float = 1.0,
                            marker: Optional[str] = None,
                            markersize: float = 3.0,
                            grid: bool = False,
                            error_bars: Optional[np.ndarray] = None,
                            save_path: Optional[str] = None,
                            dpi: int = 600) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建出版物级别的图表
    
    Parameters:
    -----------
    x, y : array-like
        数据
    figsize : tuple, optional
        图表尺寸（英寸），默认使用期刊标准尺寸
    xlabel, ylabel : str
        坐标轴标签
    title : str, optional
        标题（出版物通常不使用标题）
    style : str
        风格: 'nature', 'science', 'ieee'
    color : str
        线条颜色
    linewidth : float
        线条宽度
    marker : str, optional
        标记样式
    markersize : float
        标记大小
    grid : bool
        是否显示网格（出版物通常不显示）
    error_bars : array, optional
        误差棒数据
    save_path : str, optional
        保存路径
    dpi : int
        分辨率（出版物推荐 600+）
    
    Returns:
    --------
    fig, ax : matplotlib 图表对象
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = np.sin(x)
    >>> fig, ax = plot_publication_figure(x, y, style='nature')
    """
    # 应用期刊风格
    if style == 'nature':
        PublicationStyle.apply_nature_style()
        if figsize is None:
            figsize = PublicationStyle.NATURE_STYLE['figure.figsize']
    elif style == 'science':
        PublicationStyle.apply_science_style()
        if figsize is None:
            figsize = PublicationStyle.SCIENCE_STYLE['figure.figsize']
    elif style == 'ieee':
        PublicationStyle.apply_ieee_style()
        if figsize is None:
            figsize = PublicationStyle.IEEE_STYLE['figure.figsize']
    else:
        raise ValueError(f"Unknown style: {style}")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制数据
    if error_bars is not None:
        ax.errorbar(x, y, yerr=error_bars, 
                   color=color, linewidth=linewidth,
                   marker=marker, markersize=markersize,
                   capsize=2, capthick=0.5, elinewidth=0.5)
    else:
        ax.plot(x, y, color=color, linewidth=linewidth, 
               marker=marker, markersize=markersize)
    
    # 设置标签
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if title:
        ax.set_title(title)
    
    # 网格（通常不使用）
    if grid:
        ax.grid(True, alpha=0.3, linewidth=0.5, linestyle='--')
    
    # 优化布局
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', 
                   pad_inches=0.05, transparent=False)
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


def plot_multi_panel_figure(data_list: List[Tuple[np.ndarray, np.ndarray]],
                             nrows: int = 1,
                             ncols: int = 2,
                             style: str = 'nature',
                             figsize: Optional[Tuple[float, float]] = None,
                             sharex: bool = False,
                             sharey: bool = False,
                             labels: Optional[List[str]] = None,
                             save_path: Optional[str] = None,
                             dpi: int = 600) -> Tuple[plt.Figure, np.ndarray]:
    """
    创建多面板图表（子图）
    
    Parameters:
    -----------
    data_list : list of tuple
        每个元素是 (x, y) 数据元组
    nrows, ncols : int
        行数和列数
    style : str
        期刊风格
    figsize : tuple, optional
        整体图表尺寸
    sharex, sharey : bool
        是否共享坐标轴
    labels : list of str, optional
        子图标签 (A, B, C, ...)
    save_path : str, optional
        保存路径
    dpi : int
        分辨率
    
    Returns:
    --------
    fig, axes : matplotlib 图表对象
    
    Example:
    --------
    >>> data1 = (np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50)))
    >>> data2 = (np.linspace(0, 10, 50), np.cos(np.linspace(0, 10, 50)))
    >>> fig, axes = plot_multi_panel_figure([data1, data2], nrows=1, ncols=2)
    """
    # 应用期刊风格
    if style == 'nature':
        PublicationStyle.apply_nature_style()
        if figsize is None:
            figsize = (7.0, 2.5)  # 双栏宽度
    elif style == 'science':
        PublicationStyle.apply_science_style()
        if figsize is None:
            figsize = (7.0, 2.5)
    else:
        if figsize is None:
            figsize = (7.0, 3.0)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, 
                            sharex=sharex, sharey=sharey)
    
    # 如果只有一个子图，确保 axes 是二维数组
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1 or ncols == 1:
        axes = axes.reshape(nrows, ncols)
    
    # 绘制每个子图
    for idx, (x, y) in enumerate(data_list):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]
        
        ax.plot(x, y, linewidth=1.0)
        
        # 添加子图标签
        if labels and idx < len(labels):
            ax.text(-0.1, 1.05, labels[idx], transform=ax.transAxes,
                   fontsize=8, fontweight='bold', va='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, axes


def plot_with_annotations(x: np.ndarray,
                          y: np.ndarray,
                          annotations: List[dict],
                          style: str = 'nature',
                          color: str = '#1f77b4',
                          save_path: Optional[str] = None,
                          dpi: int = 600) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建带注释的图表（突出显示特定数据点）
    
    Parameters:
    -----------
    x, y : array-like
        数据
    annotations : list of dict
        注释列表，每个字典包含:
        - 'x': x坐标
        - 'y': y坐标
        - 'text': 注释文本
        - 'arrow': 是否使用箭头（默认True）
    style : str
        期刊风格
    color : str
        主数据颜色
    save_path : str, optional
        保存路径
    dpi : int
        分辨率
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 50)
    >>> y = np.sin(x)
    >>> annotations = [
    ...     {'x': 7.85, 'y': 1, 'text': 'Peak', 'arrow': True}
    ... ]
    >>> fig, ax = plot_with_annotations(x, y, annotations)
    """
    # 应用期刊风格
    if style == 'nature':
        PublicationStyle.apply_nature_style()
    
    fig, ax = plt.subplots()
    ax.plot(x, y, color=color, linewidth=1.0)
    
    # 添加注释
    for ann in annotations:
        x_pos = ann['x']
        y_pos = ann['y']
        text = ann.get('text', '')
        use_arrow = ann.get('arrow', True)
        
        if use_arrow:
            ax.annotate(text, xy=(x_pos, y_pos),
                       xytext=(x_pos + 0.5, y_pos + 0.2),
                       fontsize=6,
                       arrowprops=dict(arrowstyle='->', lw=0.5, color='black'))
        else:
            ax.text(x_pos, y_pos, text, fontsize=6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    
    return fig, ax


if __name__ == '__main__':
    # 测试代码
    print("测试科学出版物级图表...")
    
    # 测试1: Nature 风格图表
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    
    fig, ax = plot_publication_figure(x, y, 
                                     xlabel='Time (s)',
                                     ylabel='Amplitude',
                                     style='nature',
                                     save_path='test_nature_style.png')
    plt.close()
    
    # 测试2: 多面板图表
    data1 = (x, np.sin(x))
    data2 = (x, np.cos(x))
    data3 = (x, np.sin(x) * np.cos(x))
    data4 = (x, x**0.5)
    
    fig, axes = plot_multi_panel_figure(
        [data1, data2, data3, data4],
        nrows=2, ncols=2,
        labels=['A', 'B', 'C', 'D'],
        style='nature',
        save_path='test_multi_panel.png'
    )
    plt.close()
    
    # 测试3: 带注释的图表
    annotations = [
        {'x': np.pi/2, 'y': 1, 'text': 'Maximum', 'arrow': True},
        {'x': 3*np.pi/2, 'y': -1, 'text': 'Minimum', 'arrow': True}
    ]
    
    fig, ax = plot_with_annotations(x, y, annotations,
                                   style='nature',
                                   save_path='test_annotated.png')
    plt.close()
    
    print("所有测试完成！")
