"""
折线图模板
提供多种折线图绘制功能
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union


def plot_single_line(x, y, 
                     title: str = '',
                     xlabel: str = 'X',
                     ylabel: str = 'Y',
                     label: Optional[str] = None,
                     color: str = '#1f77b4',
                     linestyle: str = '-',
                     linewidth: float = 1.5,
                     marker: Optional[str] = None,
                     grid: bool = True,
                     save_path: Optional[str] = None,
                     dpi: int = 300,
                     figsize: tuple = (8, 5)):
    """
    绘制单条折线图
    
    Parameters:
    -----------
    x : array-like
        X轴数据
    y : array-like
        Y轴数据
    title : str
        图表标题
    xlabel : str
        X轴标签
    ylabel : str
        Y轴标签
    label : str, optional
        数据标签
    color : str
        线条颜色
    linestyle : str
        线条样式: '-', '--', '-.', ':'
    linewidth : float
        线条宽度
    marker : str, optional
        标记样式: 'o', 's', '^', 'v', 'd', '*'
    grid : bool
        是否显示网格
    save_path : str, optional
        保存路径
    dpi : int
        分辨率
    figsize : tuple
        图表大小
    
    Returns:
    --------
    fig, ax : matplotlib图表对象
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x)
    >>> plot_single_line(x, y, title='Sine Wave', save_path='sine.png')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth,
            marker=marker, label=label)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if label:
        ax.legend()
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


def plot_multi_lines(x, y_list: List,
                     labels: Optional[List[str]] = None,
                     title: str = '',
                     xlabel: str = 'X',
                     ylabel: str = 'Y',
                     colors: Optional[List[str]] = None,
                     linestyles: Optional[List[str]] = None,
                     linewidth: float = 1.5,
                     markers: Optional[List[str]] = None,
                     grid: bool = True,
                     legend_loc: str = 'best',
                     save_path: Optional[str] = None,
                     dpi: int = 300,
                     figsize: tuple = (8, 5)):
    """
    绘制多条折线图
    
    Parameters:
    -----------
    x : array-like
        X轴数据（所有线条共用）
    y_list : list of array-like
        Y轴数据列表
    labels : list of str, optional
        数据标签列表
    colors : list of str, optional
        颜色列表
    linestyles : list of str, optional
        线条样式列表
    markers : list of str, optional
        标记样式列表
    其他参数同 plot_single_line
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y1 = np.sin(x)
    >>> y2 = np.cos(x)
    >>> plot_multi_lines(x, [y1, y2], labels=['sin', 'cos'])
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_lines = len(y_list)
    
    # 默认颜色
    if colors is None:
        from ..style_config.color_schemes import ACADEMIC_COLORS
        colors = ACADEMIC_COLORS[:n_lines]
    
    # 默认标签
    if labels is None:
        labels = [f'Line {i+1}' for i in range(n_lines)]
    
    # 默认线型
    if linestyles is None:
        linestyles = ['-'] * n_lines
    
    # 默认标记
    if markers is None:
        markers = [None] * n_lines
    
    # 绘制每条线
    for i, y in enumerate(y_list):
        ax.plot(x, y, 
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=linewidth,
                marker=markers[i % len(markers)],
                label=labels[i],
                markevery=max(1, len(x) // 10))  # 每10个点显示一个标记
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


def plot_with_confidence_interval(x, y_mean, y_std,
                                   title: str = '',
                                   xlabel: str = 'X',
                                   ylabel: str = 'Y',
                                   label: str = 'Mean',
                                   color: str = '#1f77b4',
                                   alpha: float = 0.2,
                                   grid: bool = True,
                                   save_path: Optional[str] = None,
                                   dpi: int = 300,
                                   figsize: tuple = (8, 5)):
    """
    绘制带置信区间的折线图
    
    Parameters:
    -----------
    x : array-like
        X轴数据
    y_mean : array-like
        均值数据
    y_std : array-like
        标准差数据
    alpha : float
        置信区间透明度
    其他参数同 plot_single_line
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y_mean = np.sin(x)
    >>> y_std = 0.1 * np.ones_like(x)
    >>> plot_with_confidence_interval(x, y_mean, y_std)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制均值线
    ax.plot(x, y_mean, color=color, linewidth=2, label=label)
    
    # 绘制置信区间
    ax.fill_between(x, y_mean - y_std, y_mean + y_std,
                     color=color, alpha=alpha, label='±1 std')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


def plot_time_series(dates, values,
                     title: str = 'Time Series',
                     ylabel: str = 'Value',
                     color: str = '#1f77b4',
                     grid: bool = True,
                     save_path: Optional[str] = None,
                     dpi: int = 300,
                     figsize: tuple = (10, 5)):
    """
    绘制时间序列图
    
    Parameters:
    -----------
    dates : array-like
        日期数据（datetime对象）
    values : array-like
        数值数据
    其他参数同 plot_single_line
    
    Example:
    --------
    >>> import pandas as pd
    >>> dates = pd.date_range('2020-01-01', periods=100)
    >>> values = np.random.randn(100).cumsum()
    >>> plot_time_series(dates, values)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(dates, values, color=color, linewidth=1.5)
    
    ax.set_xlabel('Date')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # 自动格式化日期
    fig.autofmt_xdate()
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    
    # 测试1: 单条折线
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    plot_single_line(x, y, title='Single Line', xlabel='x', ylabel='sin(x)',
                     marker='o', save_path='test_single_line.png')
    
    # 测试2: 多条折线
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    plot_multi_lines(x, [y1, y2, y3], 
                     labels=['sin(x)', 'cos(x)', 'sin(x)cos(x)'],
                     title='Multiple Lines',
                     save_path='test_multi_lines.png')
    
    # 测试3: 带置信区间
    y_mean = np.sin(x)
    y_std = 0.2 * np.abs(np.sin(x))
    plot_with_confidence_interval(x, y_mean, y_std,
                                   title='Line with Confidence Interval',
                                   save_path='test_confidence.png')
    
    print("测试完成！")
