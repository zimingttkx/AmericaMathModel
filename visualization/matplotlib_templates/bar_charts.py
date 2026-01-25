"""
柱状图模板
提供多种柱状图绘制功能
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Union


def plot_bar_chart(categories: List[str], values: List[float],
                   title: str = '',
                   xlabel: str = '',
                   ylabel: str = 'Value',
                   color: str = '#1f77b4',
                   width: float = 0.6,
                   grid: bool = True,
                   save_path: Optional[str] = None,
                   dpi: int = 300,
                   figsize: tuple = (8, 5)):
    """
    绘制简单柱状图
    
    Parameters:
    -----------
    categories : list of str
        类别标签
    values : list of float
        数值
    title : str
        图表标题
    xlabel : str
        X轴标签
    ylabel : str
        Y轴标签
    color : str
        柱子颜色
    width : float
        柱子宽度
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
    >>> categories = ['A', 'B', 'C', 'D']
    >>> values = [23, 45, 56, 78]
    >>> plot_bar_chart(categories, values, title='Simple Bar Chart')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_pos = np.arange(len(categories))
    ax.bar(x_pos, values, width=width, color=color, alpha=0.8)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


def plot_grouped_bar_chart(categories: List[str],
                           values_list: List[List[float]],
                           labels: List[str],
                           title: str = '',
                           xlabel: str = '',
                           ylabel: str = 'Value',
                           colors: Optional[List[str]] = None,
                           width: float = 0.25,
                           grid: bool = True,
                           legend_loc: str = 'best',
                           save_path: Optional[str] = None,
                           dpi: int = 300,
                           figsize: tuple = (10, 5)):
    """
    绘制分组柱状图
    
    Parameters:
    -----------
    categories : list of str
        类别标签
    values_list : list of list
        多组数值数据
    labels : list of str
        每组数据的标签
    colors : list of str, optional
        颜色列表
    width : float
        柱子宽度
    其他参数同 plot_bar_chart
    
    Example:
    --------
    >>> categories = ['A', 'B', 'C', 'D']
    >>> values1 = [23, 45, 56, 78]
    >>> values2 = [34, 55, 44, 88]
    >>> plot_grouped_bar_chart(categories, [values1, values2], 
    ...                        labels=['Group 1', 'Group 2'])
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_groups = len(categories)
    n_bars = len(values_list)
    
    # 默认颜色
    if colors is None:
        from ..style_config.color_schemes import ACADEMIC_COLORS
        colors = ACADEMIC_COLORS[:n_bars]
    
    x_pos = np.arange(n_groups)
    
    # 绘制每组柱子
    for i, (values, label) in enumerate(zip(values_list, labels)):
        offset = (i - n_bars/2 + 0.5) * width
        ax.bar(x_pos + offset, values, width=width, 
               color=colors[i], alpha=0.8, label=label)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


def plot_stacked_bar_chart(categories: List[str],
                           values_list: List[List[float]],
                           labels: List[str],
                           title: str = '',
                           xlabel: str = '',
                           ylabel: str = 'Value',
                           colors: Optional[List[str]] = None,
                           width: float = 0.6,
                           grid: bool = True,
                           legend_loc: str = 'best',
                           save_path: Optional[str] = None,
                           dpi: int = 300,
                           figsize: tuple = (8, 5)):
    """
    绘制堆叠柱状图
    
    Parameters:
    -----------
    categories : list of str
        类别标签
    values_list : list of list
        多组数值数据
    labels : list of str
        每组数据的标签
    其他参数同 plot_grouped_bar_chart
    
    Example:
    --------
    >>> categories = ['A', 'B', 'C', 'D']
    >>> values1 = [23, 45, 56, 78]
    >>> values2 = [34, 55, 44, 88]
    >>> plot_stacked_bar_chart(categories, [values1, values2],
    ...                        labels=['Part 1', 'Part 2'])
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 默认颜色
    if colors is None:
        from ..style_config.color_schemes import ACADEMIC_COLORS
        colors = ACADEMIC_COLORS[:len(values_list)]
    
    x_pos = np.arange(len(categories))
    bottom = np.zeros(len(categories))
    
    # 绘制堆叠柱子
    for i, (values, label) in enumerate(zip(values_list, labels)):
        ax.bar(x_pos, values, width=width, bottom=bottom,
               color=colors[i], alpha=0.8, label=label)
        bottom += np.array(values)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc=legend_loc)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


def plot_horizontal_bar_chart(categories: List[str], values: List[float],
                              title: str = '',
                              xlabel: str = 'Value',
                              ylabel: str = '',
                              color: str = '#1f77b4',
                              height: float = 0.6,
                              grid: bool = True,
                              save_path: Optional[str] = None,
                              dpi: int = 300,
                              figsize: tuple = (8, 6)):
    """
    绘制水平柱状图
    
    Parameters:
    -----------
    categories : list of str
        类别标签
    values : list of float
        数值
    height : float
        柱子高度
    其他参数同 plot_bar_chart
    
    Example:
    --------
    >>> categories = ['Category A', 'Category B', 'Category C']
    >>> values = [23, 45, 56]
    >>> plot_horizontal_bar_chart(categories, values)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(categories))
    ax.barh(y_pos, values, height=height, color=color, alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    return fig, ax


if __name__ == '__main__':
    # 测试代码
    
    # 测试1: 简单柱状图
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [23, 45, 56, 78, 32]
    plot_bar_chart(categories, values, title='Simple Bar Chart',
                   save_path='test_bar_chart.png')
    
    # 测试2: 分组柱状图
    values1 = [23, 45, 56, 78, 32]
    values2 = [34, 55, 44, 88, 42]
    values3 = [28, 38, 48, 68, 38]
    plot_grouped_bar_chart(categories, [values1, values2, values3],
                          labels=['Method 1', 'Method 2', 'Method 3'],
                          title='Grouped Bar Chart',
                          save_path='test_grouped_bar.png')
    
    # 测试3: 堆叠柱状图
    plot_stacked_bar_chart(categories, [values1, values2, values3],
                          labels=['Part 1', 'Part 2', 'Part 3'],
                          title='Stacked Bar Chart',
                          save_path='test_stacked_bar.png')
    
    # 测试4: 水平柱状图
    plot_horizontal_bar_chart(categories, values,
                             title='Horizontal Bar Chart',
                             save_path='test_horizontal_bar.png')
    
    print("测试完成！")
