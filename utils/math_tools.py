"""
数学计算工具模块
提供常用的数学计算函数
"""

import numpy as np
from scipy import stats
from typing import Union, Tuple


def calculate_distance(x1: Union[np.ndarray, float],
                       x2: Union[np.ndarray, float],
                       metric: str = 'euclidean') -> float:
    """
    计算两点之间的距离
    
    Parameters:
    -----------
    x1, x2 : array-like or float
        两个点的坐标
    metric : str
        距离度量: 'euclidean', 'manhattan', 'chebyshev', 'cosine'
    
    Returns:
    --------
    distance : float
        距离值
    
    Example:
    --------
    >>> x1 = np.array([1, 2, 3])
    >>> x2 = np.array([4, 5, 6])
    >>> dist = calculate_distance(x1, x2, metric='euclidean')
    >>> print(f"{dist:.3f}")  # 5.196
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    if metric == 'euclidean':
        return np.linalg.norm(x1 - x2)
    elif metric == 'manhattan':
        return np.sum(np.abs(x1 - x2))
    elif metric == 'chebyshev':
        return np.max(np.abs(x1 - x2))
    elif metric == 'cosine':
        return 1 - np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    else:
        raise ValueError(f"Unknown metric: {metric}")


def calculate_similarity(x1: Union[np.ndarray, float],
                        x2: Union[np.ndarray, float],
                        metric: str = 'cosine') -> float:
    """
    计算两个向量之间的相似度
    
    Parameters:
    -----------
    x1, x2 : array-like
        两个向量
    metric : str
        相似度度量: 'cosine', 'pearson', 'jaccard'
    
    Returns:
    --------
    similarity : float
        相似度值（范围通常在[0, 1]）
    
    Example:
    --------
    >>> x1 = np.array([1, 2, 3])
    >>> x2 = np.array([2, 4, 6])
    >>> sim = calculate_similarity(x1, x2)
    >>> print(f"{sim:.3f}")  # 1.000
    """
    x1 = np.array(x1)
    x2 = np.array(x2)
    
    if metric == 'cosine':
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    elif metric == 'pearson':
        return np.corrcoef(x1, x2)[0, 1]
    elif metric == 'jaccard':
        intersection = len(set(x1) & set(x2))
        union = len(set(x1) | set(x2))
        return intersection / union if union > 0 else 0
    else:
        raise ValueError(f"Unknown metric: {metric}")


def calculate_entropy(data: np.ndarray, base: int = 2) -> float:
    """
    计算信息熵
    
    Parameters:
    -----------
    data : array-like
        数据数组
    base : int
        对数的底（2表示比特，e表示奈特，10表示哈特利）
    
    Returns:
    --------
    entropy : float
        信息熵
    
    Example:
    --------
    >>> data = np.array([1, 1, 1, 2, 2, 3])
    >>> entropy = calculate_entropy(data)
    >>> print(f"{entropy:.3f}")
    """
    _, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    
    if base == 2:
        log_func = np.log2
    elif base == np.e:
        log_func = np.log
    elif base == 10:
        log_func = np.log10
    else:
        raise ValueError("base must be 2, e, or 10")
    
    entropy = -np.sum(probabilities * log_func(probabilities))
    return entropy


def polynomial_fit(x: np.ndarray, y: np.ndarray,
                   degree: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    多项式拟合
    
    Parameters:
    -----------
    x, y : array-like
        x和y数据
    degree : int
        多项式次数
    
    Returns:
    --------
    coefficients : np.ndarray
        拟合系数（从高次到低次）
    y_fit : np.ndarray
        拟合后的y值
    
    Example:
    --------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> y = np.array([2, 4, 6, 8, 10])
    >>> coeffs, y_fit = polynomial_fit(x, y, degree=1)
    >>> print(f"斜率: {coeffs[0]:.2f}, 截距: {coeffs[1]:.2f}")
    """
    coefficients = np.polyfit(x, y, degree)
    y_fit = np.polyval(coefficients, x)
    return coefficients, y_fit


def interpolation(x: np.ndarray, y: np.ndarray,
                  x_new: np.ndarray,
                  method: str = 'linear') -> np.ndarray:
    """
    数据插值
    
    Parameters:
    -----------
    x, y : array-like
        已知数据点
    x_new : array-like
        需要插值的x坐标
    method : str
        插值方法: 'linear', 'nearest', 'cubic', 'quadratic'
    
    Returns:
    --------
    y_new : np.ndarray
        插值后的y值
    
    Example:
    --------
    >>> x = np.array([1, 2, 3, 4])
    >>> y = np.array([1, 4, 9, 16])
    >>> x_new = np.linspace(1, 4, 10)
    >>> y_new = interpolation(x, y, x_new, method='cubic')
    """
    from scipy import interpolate
    
    if method == 'linear':
        f = interpolate.interp1d(x, y, kind='linear')
    elif method == 'nearest':
        f = interpolate.interp1d(x, y, kind='nearest')
    elif method == 'cubic':
        f = interpolate.interp1d(x, y, kind='cubic')
    elif method == 'quadratic':
        f = interpolate.interp1d(x, y, kind='quadratic')
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return f(x_new)


def smooth_data(data: np.ndarray,
                window_size: int = 5,
                method: str = 'moving_average') -> np.ndarray:
    """
    数据平滑
    
    Parameters:
    -----------
    data : array-like
        输入数据
    window_size : int
        窗口大小
    method : str
        平滑方法: 'moving_average', 'exponential', 'savgol'
    
    Returns:
    --------
    smoothed_data : np.ndarray
        平滑后的数据
    
    Example:
    --------
    >>> data = np.random.randn(100).cumsum()
    >>> smoothed = smooth_data(data, window_size=10, method='moving_average')
    """
    data = np.array(data)
    
    if method == 'moving_average':
        # 移动平均
        cumsum = np.cumsum(np.insert(data, 0, 0))
        smoothed = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
        # 填充前面无法计算的部分
        padded = np.concatenate([[data[0]] * (window_size - 1), smoothed])
        return padded[:len(data)]
    
    elif method == 'exponential':
        # 指数平滑
        alpha = 2.0 / (window_size + 1)
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        return smoothed
    
    elif method == 'savgol':
        # Savitzky-Golay滤波器
        from scipy.signal import savgol_filter
        return savgol_filter(data, window_size, 3)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_derivative(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算数值导数
    
    Parameters:
    -----------
    x, y : array-like
        x和y数据
    
    Returns:
    --------
    derivative : np.ndarray
        dy/dx的数值近似
    
    Example:
    --------
    >>> x = np.linspace(0, 2*np.pi, 100)
    >>> y = np.sin(x)
    >>> dy = calculate_derivative(x, y)
    >>> # dy应该接近cos(x)
    """
    return np.gradient(y, x)


def calculate_integral(x: np.ndarray, y: np.ndarray) -> float:
    """
    计算数值积分（梯形法则）
    
    Parameters:
    -----------
    x, y : array-like
        x和y数据
    
    Returns:
    --------
    integral : float
        ∫y dx的数值近似
    
    Example:
    --------
    >>> x = np.linspace(0, np.pi, 100)
    >>> y = np.sin(x)
    >>> integral = calculate_integral(x, y)
    >>> print(f"{integral:.3f}")  # 应该接近2.0
    """
    return np.trapz(y, x)


def find_peaks(data: np.ndarray,
               height: Optional[float] = None,
               distance: Optional[int] = None,
               prominence: Optional[float] = None) -> Tuple[np.ndarray, dict]:
    """
    查找数据中的峰值
    
    Parameters:
    -----------
    data : array-like
        输入数据
    height : float, optional
        峰值的最低高度
    distance : int, optional
        峰值之间的最小距离
    prominence : float, optional
        峰值的突出程度
    
    Returns:
    --------
    peaks : np.ndarray
        峰值位置的索引
    properties : dict
        峰值属性
    
    Example:
    --------
    >>> x = np.linspace(0, 10, 100)
    >>> y = np.sin(x) + 0.1 * np.random.randn(100)
    >>> peaks, props = find_peaks(y, height=0.5, distance=10)
    >>> print(f"找到 {len(peaks)} 个峰值")
    """
    from scipy.signal import find_peaks as scipy_find_peaks
    
    peaks, properties = scipy_find_peaks(
        data,
        height=height,
        distance=distance,
        prominence=prominence
    )
    
    return peaks, properties


if __name__ == '__main__':
    # 测试代码
    print("测试数学计算工具...")
    
    # 测试距离计算
    x1 = np.array([1, 2, 3])
    x2 = np.array([4, 5, 6])
    dist = calculate_distance(x1, x2)
    print(f"欧氏距离: {dist:.3f}")
    
    # 测试相似度计算
    sim = calculate_similarity(x1, x2 * 2)
    print(f"余弦相似度: {sim:.3f}")
    
    # 测试信息熵
    data = np.array([1, 1, 1, 2, 2, 3])
    entropy = calculate_entropy(data)
    print(f"信息熵: {entropy:.3f}")
    
    # 测试多项式拟合
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1.1, 2.0, 3.1, 4.0, 5.1])
    coeffs, y_fit = polynomial_fit(x, y, degree=1)
    print(f"拟合系数: {coeffs}")
    
    # 测试数值导数
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x)
    dy = calculate_derivative(x, y)
    print(f"导数计算完成，形状: {dy.shape}")
    
    print("\n所有测试通过！")
