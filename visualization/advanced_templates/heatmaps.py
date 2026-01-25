"""
热力图模板
提供论文级热力图可视化
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Optional, List, Tuple, Union


def plot_heatmap(data: Union[np.ndarray, pd.DataFrame],
                vmin: Optional[float] = None,
                vmax: Optional[float] = None,
                cmap: str = 'RdYlBu_r',
                center: Optional[float] = None,
                annot: bool = True,
                fmt: str = '.2f',
                linewidths: float = 0.5,
                linecolor: str = 'white',
                cbar: bool = True,
                xticklabels: Optional[List[str]] = None,
                yticklabels: Optional[List[str]] = None,
                xlabel: Optional[str] = None,
                ylabel: Optional[str] = None,
                title: Optional[str] = None,
                figsize: Tuple[float, float] = (5, 4),
                save_path: Optional[str] = None,
                dpi: int = 300) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制热力图
    
    Parameters:
    -----------
    data : DataFrame or array-like
        数据矩阵
    vmin, vmax : float, optional
        颜色范围的最小值和最大值
    cmap : str
        颜色映射
    center : float, optional
        颜色映射的中心值
    annot : bool
        是否显示数值注释
    fmt : str
        数值格式
    linewidths : float
        单元格间隔线宽度
    linecolor : str
        间隔线颜色
    cbar : bool
        是否显示颜色条
    xticklabels, yticklabels : list, optional
        坐标轴标签
    xlabel, ylabel : str, optional
        坐标轴标题
    title : str, optional
        图表标题
    figsize : tuple
        图表尺寸
    save_path : str, optional
        保存路径
    dpi : int
        分辨率
    
    Returns:
    --------
    fig, ax : matplotlib 图表对象
    
    Example:
    --------
    >>> data = np.random.rand(5, 5)
    >>> fig, ax = plot_heatmap(data, cmap='RdYlBu_r')
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 如果是 DataFrame，使用其索引和列
    if isinstance(data, pd.DataFrame):
        if xticklabels is None:
            xticklabels = data.columns
        if yticklabels is None:
            yticklabels = data.index
        data = data.values
    
    # 绘制热力图
    sns.heatmap(data, vmin=vmin, vmax=vmax, cmap=cmap, center=center,
                annot=annot, fmt=fmt, linewidths=linewidths, linecolor=linecolor,
                cbar=cbar, xticklabels=xticklabels, yticklabels=yticklabels,
                ax=ax)
    
    # 设置标签
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"热力图已保存至: {save_path}")
    
    return fig, ax


def plot_correlation_matrix(data: Union[pd.DataFrame, np.ndarray],
                           variables: Optional[List[str]] = None,
                           method: str = 'pearson',
                           cmap: str = 'RdYlBu_r',
                           annot: bool = True,
                           fmt: str = '.2f',
                           figsize: Tuple[float, float] = (6, 5),
                           save_path: Optional[str] = None,
                           dpi: int = 300) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制相关性矩阵热力图
    
    Parameters:
    -----------
    data : DataFrame or array-like
        数据
    variables : list of str, optional
        变量名称
    method : str
        相关性方法: 'pearson', 'spearman', 'kendall'
    cmap : str
        颜色映射
    annot : bool
        是否显示相关系数
    fmt : str
        数值格式
    figsize : tuple
        图表尺寸
    save_path : str, optional
        保存路径
    dpi : int
        分辨率
    
    Returns:
    --------
    fig, ax : matplotlib 图表对象
    
    Example:
    --------
    >>> df = pd.DataFrame(np.random.rand(100, 4), columns=['A', 'B', 'C', 'D'])
    >>> fig, ax = plot_correlation_matrix(df)
    """
    # 计算相关性矩阵
    if isinstance(data, pd.DataFrame):
        corr_matrix = data.corr(method=method)
        if variables is None:
            variables = data.columns.tolist()
    else:
        data = pd.DataFrame(data)
        corr_matrix = data.corr(method=method)
        if variables is None:
            variables = [f'Var{i+1}' for i in range(data.shape[1])]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制热力图
    sns.heatmap(corr_matrix, cmap=cmap, annot=annot, fmt=fmt,
                vmin=-1, vmax=1, center=0,
                linewidths=0.5, linecolor='white',
                xticklabels=variables, yticklabels=variables,
                square=True, ax=ax)
    
    ax.set_title(f'{method.capitalize()} Correlation Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"相关性矩阵已保存至: {save_path}")
    
    return fig, ax


def plot_clustermap(data: Union[pd.DataFrame, np.ndarray],
                   method: str = 'average',
                   metric: str = 'euclidean',
                   cmap: str = 'RdYlBu_r',
                   standard_scale: Optional[int] = None,
                   figsize: Tuple[float, float] = (8, 6),
                   save_path: Optional[str] = None,
                   dpi: int = 300):
    """
    绘制聚类热力图（带层次聚类）
    
    Parameters:
    -----------
    data : DataFrame or array-like
        数据矩阵
    method : str
        连接方法: 'single', 'complete', 'average', 'ward'
    metric : str
        距离度量: 'euclidean', 'correlation', 'cityblock'
    cmap : str
        颜色映射
    standard_scale : int, optional
        标准化: 0=按行，1=按列
    figsize : tuple
        图表尺寸
    save_path : str, optional
        保存路径
    dpi : int
        分辨率
    
    Returns:
    --------
    g : seaborn ClusterGrid 对象
    
    Example:
    --------
    >>> data = np.random.rand(10, 5)
    >>> g = plot_clustermap(data, method='average')
    """
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    g = sns.clustermap(data, method=method, metric=metric,
                      cmap=cmap, standard_scale=standard_scale,
                      figsize=figsize, linewidths=0.5,
                      linecolor='white')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"聚类热力图已保存至: {save_path}")
    
    return g


def plot_imshow(data: Union[np.ndarray, pd.DataFrame],
               extent: Optional[Tuple[float, float, float, float]] = None,
               cmap: str = 'viridis',
               colorbar: bool = True,
               aspect: str = 'auto',
               xlabel: Optional[str] = None,
               ylabel: Optional[str] = None,
               title: Optional[str] = None,
               figsize: Tuple[float, float] = (6, 5),
               save_path: Optional[str] = None,
               dpi: int = 300) -> Tuple[plt.Figure, plt.Axes]:
    """
    绘制图像式热力图（适用于连续数据）
    
    Parameters:
    -----------
    data : array-like
        2D 数据数组
    extent : tuple, optional
        坐标范围 [xmin, xmax, ymin, ymax]
    cmap : str
        颜色映射
    colorbar : bool
        是否显示颜色条
    aspect : str
        纵横比: 'auto', 'equal'
    xlabel, ylabel : str, optional
        坐标轴标题
    title : str, optional
        图表标题
    figsize : tuple
        图表尺寸
    save_path : str, optional
        保存路径
    dpi : int
        分辨率
    
    Returns:
    --------
    fig, ax : matplotlib 图表对象
    
    Example:
    --------
    >>> x = np.linspace(-5, 5, 100)
    >>> y = np.linspace(-5, 5, 100)
    >>> X, Y = np.meshgrid(x, y)
    >>> Z = np.sin(np.sqrt(X**2 + Y**2))
    >>> fig, ax = plot_imshow(Z, extent=[-5, 5, -5, 5])
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(data, cmap=cmap, extent=extent, aspect=aspect, origin='lower')
    
    if colorbar:
        plt.colorbar(im, ax=ax)
    
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    return fig, ax


if __name__ == '__main__':
    # 测试代码
    print("测试热力图模板...")
    
    # 测试1: 基本热力图
    data = np.random.rand(5, 5)
    fig, ax = plot_heatmap(data, cmap='RdYlBu_r',
                          title='Test Heatmap',
                          save_path='test_heatmap.png')
    plt.close()
    
    # 测试2: 相关性矩阵
    df = pd.DataFrame(np.random.rand(100, 4), 
                     columns=['Variable A', 'Variable B', 'Variable C', 'Variable D'])
    fig, ax = plot_correlation_matrix(df, save_path='test_correlation.png')
    plt.close()
    
    # 测试3: 聚类热力图
    cluster_data = np.random.rand(10, 5)
    g = plot_clustermap(cluster_data, save_path='test_clustermap.png')
    plt.close()
    
    # 测试4: 图像式热力图
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fig, ax = plot_imshow(Z, extent=[-5, 5, -5, 5],
                         xlabel='X', ylabel='Y',
                         title='sin(sqrt(X^2 + Y^2))',
                         save_path='test_imshow.png')
    plt.close()
    
    print("所有测试完成！")
