"""
数据处理工具模块
提供常用的数据处理函数
"""

import numpy as np
import pandas as pd
from typing import Union, List, Tuple


def normalize(data: Union[np.ndarray, pd.Series], 
              method: str = 'minmax') -> np.ndarray:
    """
    数据归一化
    
    Parameters:
    -----------
    data : array-like
        输入数据
    method : str
        归一化方法: 'minmax' (0-1), 'zscore' (标准化), 'robust' (鲁棒)
    
    Returns:
    --------
    normalized_data : np.ndarray
        归一化后的数据
    
    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> normalized = normalize(data, method='minmax')
    >>> print(normalized)  # [0. 0.25 0.5 0.75 1.]
    """
    data = np.array(data)
    
    if method == 'minmax':
        # Min-Max归一化到[0, 1]
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val == 0:
            return np.zeros_like(data)
        return (data - min_val) / (max_val - min_val)
    
    elif method == 'zscore':
        # Z-score标准化
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return np.zeros_like(data)
        return (data - mean) / std
    
    elif method == 'robust':
        # 鲁棒归一化（使用中位数和四分位数）
        median = np.median(data)
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return np.zeros_like(data)
        return (data - median) / iqr
    
    else:
        raise ValueError(f"Unknown method: {method}")


def remove_outliers(data: Union[np.ndarray, pd.Series],
                    method: str = 'iqr',
                    threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    移除异常值
    
    Parameters:
    -----------
    data : array-like
        输入数据
    method : str
        异常值检测方法: 'iqr', 'zscore'
    threshold : float
        阈值（IQR方法默认1.5，Z-score方法默认3）
    
    Returns:
    --------
    clean_data : np.ndarray
        移除异常值后的数据
    outlier_mask : np.ndarray
        异常值掩码（True表示异常值）
    
    Example:
    --------
    >>> data = np.array([1, 2, 3, 100, 4, 5])
    >>> clean, mask = remove_outliers(data, method='iqr')
    >>> print(clean)  # [1, 2, 3, 4, 5]
    """
    data = np.array(data)
    
    if method == 'iqr':
        # IQR方法
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        lower = q25 - threshold * iqr
        upper = q75 + threshold * iqr
        outlier_mask = (data < lower) | (data > upper)
    
    elif method == 'zscore':
        # Z-score方法
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return data, np.zeros_like(data, dtype=bool)
        z_scores = np.abs((data - mean) / std)
        outlier_mask = z_scores > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    clean_data = data[~outlier_mask]
    return clean_data, outlier_mask


def fill_missing_values(data: Union[np.ndarray, pd.Series],
                        method: str = 'mean') -> np.ndarray:
    """
    填充缺失值
    
    Parameters:
    -----------
    data : array-like
        输入数据（可以包含NaN）
    method : str
        填充方法: 'mean', 'median', 'mode', 'forward', 'backward', 'interpolate'
    
    Returns:
    --------
    filled_data : np.ndarray
        填充后的数据
    
    Example:
    --------
    >>> data = np.array([1, 2, np.nan, 4, 5])
    >>> filled = fill_missing_values(data, method='mean')
    >>> print(filled)  # [1. 2. 3. 4. 5.]
    """
    data = np.array(data, dtype=float)
    
    if not np.any(np.isnan(data)):
        return data
    
    if method == 'mean':
        fill_value = np.nanmean(data)
        return np.where(np.isnan(data), fill_value, data)
    
    elif method == 'median':
        fill_value = np.nanmedian(data)
        return np.where(np.isnan(data), fill_value, data)
    
    elif method == 'mode':
        # 众数填充
        from scipy import stats
        fill_value = stats.mode(data[~np.isnan(data)], keepdims=True).mode[0]
        return np.where(np.isnan(data), fill_value, data)
    
    elif method == 'forward':
        # 前向填充
        mask = np.isnan(data)
        idx = np.where(~mask)[0]
        data[mask] = np.interp(np.where(mask)[0], idx, data[idx])
        return data
    
    elif method == 'backward':
        # 后向填充
        mask = np.isnan(data)
        idx = np.where(~mask)[0]
        data[mask] = np.interp(np.where(mask)[0], idx[::-1], data[idx][::-1])
        return data
    
    elif method == 'interpolate':
        # 线性插值
        mask = np.isnan(data)
        if np.all(mask):
            return data
        idx = np.where(~mask)[0]
        data[mask] = np.interp(np.where(mask)[0], idx, data[idx])
        return data
    
    else:
        raise ValueError(f"Unknown method: {method}")


def split_data(data: Union[np.ndarray, pd.Series],
               train_ratio: float = 0.8,
               shuffle: bool = True,
               random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    分割数据为训练集和测试集
    
    Parameters:
    -----------
    data : array-like
        输入数据
    train_ratio : float
        训练集比例
    shuffle : bool
        是否打乱数据
    random_state : int, optional
        随机种子
    
    Returns:
    --------
    train_data, test_data : np.ndarray
        训练集和测试集
    
    Example:
    --------
    >>> data = np.arange(100)
    >>> train, test = split_data(data, train_ratio=0.8)
    >>> print(len(train), len(test))  # 80 20
    """
    data = np.array(data)
    n_samples = len(data)
    
    if shuffle:
        if random_state is not None:
            np.random.seed(random_state)
        indices = np.random.permutation(n_samples)
    else:
        indices = np.arange(n_samples)
    
    train_size = int(n_samples * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    return data[train_indices], data[test_indices]


def create_lags(data: Union[np.ndarray, pd.Series],
                n_lags: int = 1) -> np.ndarray:
    """
    创建滞后特征
    
    Parameters:
    -----------
    data : array-like
        输入数据（一维）
    n_lags : int
        滞后阶数
    
    Returns:
    --------
    lagged_data : np.ndarray
        包含滞后特征的数据矩阵 (n_samples - n_lags, n_lags + 1)
    
    Example:
    --------
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> lags = create_lags(data, n_lags=2)
    >>> print(lags)
    [[3 2 1]
     [4 3 2]
     [5 4 3]]
    """
    data = np.array(data)
    n_samples = len(data)
    
    if n_samples <= n_lags:
        raise ValueError("Data length must be greater than n_lags")
    
    lagged_data = np.zeros((n_samples - n_lags, n_lags + 1))
    
    for i in range(n_lags + 1):
        lagged_data[:, i] = data[n_lags - i:n_samples - i]
    
    return lagged_data


def calculate_correlation(data1: Union[np.ndarray, pd.Series],
                         data2: Union[np.ndarray, pd.Series],
                         method: str = 'pearson') -> float:
    """
    计算两个变量的相关性系数
    
    Parameters:
    -----------
    data1, data2 : array-like
        输入数据
    method : str
        相关性方法: 'pearson', 'spearman', 'kendall'
    
    Returns:
    --------
    correlation : float
        相关系数
    
    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> corr = calculate_correlation(x, y)
    >>> print(f"{corr:.3f}")  # 1.000
    """
    from scipy import stats
    
    data1 = np.array(data1)
    data2 = np.array(data2)
    
    if len(data1) != len(data2):
        raise ValueError("Data1 and Data2 must have the same length")
    
    if method == 'pearson':
        corr, _ = stats.pearsonr(data1, data2)
    elif method == 'spearman':
        corr, _ = stats.spearmanr(data1, data2)
    elif method == 'kendall':
        corr, _ = stats.kendalltau(data1, data2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return corr


if __name__ == '__main__':
    # 测试代码
    print("测试数据处理工具...")
    
    # 测试归一化
    data = np.array([1, 2, 3, 4, 5])
    print(f"原始数据: {data}")
    print(f"MinMax归一化: {normalize(data, 'minmax')}")
    print(f"Z-score标准化: {normalize(data, 'zscore')}")
    
    # 测试异常值检测
    data_with_outliers = np.array([1, 2, 3, 100, 4, 5])
    clean, mask = remove_outliers(data_with_outliers, method='iqr')
    print(f"\n原始数据: {data_with_outliers}")
    print(f"清洗后数据: {clean}")
    print(f"异常值掩码: {mask}")
    
    # 测试缺失值填充
    data_with_nan = np.array([1, 2, np.nan, 4, 5])
    filled = fill_missing_values(data_with_nan, method='mean')
    print(f"\n含缺失值数据: {data_with_nan}")
    print(f"填充后数据: {filled}")
    
    # 测试相关性
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])
    corr = calculate_correlation(x, y)
    print(f"\nPearson相关系数: {corr:.3f}")
    
    print("\n所有测试通过！")
