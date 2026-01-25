"""
文件读写工具模块
提供便捷的文件读写功能
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Union, Optional


def read_csv(filepath: str,
             encoding: str = 'utf-8',
             **kwargs) -> pd.DataFrame:
    """
    读取CSV文件
    
    Parameters:
    -----------
    filepath : str
        文件路径
    encoding : str
        文件编码
    **kwargs : dict
        传递给pd.read_csv的其他参数
    
    Returns:
    --------
    df : pd.DataFrame
        读取的数据框
    
    Example:
    --------
    >>> df = read_csv('data.csv')
    >>> print(df.head())
    """
    try:
        df = pd.read_csv(filepath, encoding=encoding, **kwargs)
        print(f"成功读取文件: {filepath}")
        print(f"数据形状: {df.shape}")
        return df
    except UnicodeDecodeError:
        # 尝试其他编码
        for enc in ['gbk', 'gb2312', 'latin1']:
            try:
                df = pd.read_csv(filepath, encoding=enc, **kwargs)
                print(f"使用编码 {enc} 成功读取文件: {filepath}")
                return df
            except:
                continue
        raise ValueError(f"无法读取文件 {filepath}，请检查文件编码")


def write_csv(data: Union[pd.DataFrame, np.ndarray],
              filepath: str,
              encoding: str = 'utf-8',
              index: bool = False,
              **kwargs) -> None:
    """
    写入CSV文件
    
    Parameters:
    -----------
    data : DataFrame or ndarray
        要保存的数据
    filepath : str
        保存路径
    encoding : str
        文件编码
    index : bool
        是否保存索引
    **kwargs : dict
        传递给to_csv的其他参数
    
    Example:
    --------
    >>> write_csv(df, 'output.csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    
    data.to_csv(filepath, encoding=encoding, index=index, **kwargs)
    print(f"数据已保存至: {filepath}")


def read_excel(filepath: str,
               sheet_name: Union[str, int] = 0,
               **kwargs) -> pd.DataFrame:
    """
    读取Excel文件
    
    Parameters:
    -----------
    filepath : str
        文件路径
    sheet_name : str or int
        工作表名称或索引
    **kwargs : dict
        传递给pd.read_excel的其他参数
    
    Returns:
    --------
    df : pd.DataFrame
        读取的数据框
    
    Example:
    --------
    >>> df = read_excel('data.xlsx', sheet_name='Sheet1')
    """
    df = pd.read_excel(filepath, sheet_name=sheet_name, **kwargs)
    print(f"成功读取Excel文件: {filepath}, 工作表: {sheet_name}")
    print(f"数据形状: {df.shape}")
    return df


def write_excel(data: Union[pd.DataFrame, dict],
                filepath: str,
                sheet_name: str = 'Sheet1',
                index: bool = False,
                **kwargs) -> None:
    """
    写入Excel文件
    
    Parameters:
    -----------
    data : DataFrame or dict
        要保存的数据（如果是dict，保存多个工作表）
    filepath : str
        保存路径
    sheet_name : str
        工作表名称
    index : bool
        是否保存索引
    **kwargs : dict
        传递给to_excel的其他参数
    
    Example:
    --------
    >>> # 单个工作表
    >>> write_excel(df, 'output.xlsx')
    >>> 
    >>> # 多个工作表
    >>> data = {'Sheet1': df1, 'Sheet2': df2}
    >>> write_excel(data, 'output.xlsx')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(data, dict):
        # 多个工作表
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            for sheet, df in data.items():
                df.to_excel(writer, sheet_name=sheet, index=index, **kwargs)
        print(f"数据已保存至: {filepath} ({len(data)} 个工作表)")
    else:
        # 单个工作表
        data.to_excel(filepath, sheet_name=sheet_name, index=index, **kwargs)
        print(f"数据已保存至: {filepath}")


def read_json(filepath: str, **kwargs) -> dict:
    """
    读取JSON文件
    
    Parameters:
    -----------
    filepath : str
        文件路径
    **kwargs : dict
        传递给json.load的其他参数
    
    Returns:
    --------
    data : dict
        JSON数据
    
    Example:
    --------
    >>> data = read_json('config.json')
    >>> print(data)
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f, **kwargs)
    print(f"成功读取JSON文件: {filepath}")
    return data


def write_json(data: dict,
               filepath: str,
               indent: int = 4,
               ensure_ascii: bool = False,
               **kwargs) -> None:
    """
    写入JSON文件
    
    Parameters:
    -----------
    data : dict
        要保存的数据
    filepath : str
        保存路径
    indent : int
        缩进空格数
    ensure_ascii : bool
        是否确保ASCII编码
    **kwargs : dict
        传递给json.dump的其他参数
    
    Example:
    --------
    >>> data = {'name': 'test', 'value': 123}
    >>> write_json(data, 'config.json')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, **kwargs)
    print(f"数据已保存至: {filepath}")


def load_data(filepath: str, **kwargs) -> pd.DataFrame:
    """
    自动识别文件格式并加载数据
    
    Parameters:
    -----------
    filepath : str
        文件路径
    **kwargs : dict
        传递给读取函数的其他参数
    
    Returns:
    --------
    df : pd.DataFrame
        读取的数据框
    
    Supported formats:
    ------------------
    - .csv
    - .xlsx, .xls
    - .json (会尝试转换为DataFrame)
    - .txt (作为CSV读取)
    
    Example:
    --------
    >>> df = load_data('data.csv')  # 自动识别CSV
    >>> df = load_data('data.xlsx') # 自动识别Excel
    """
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix == '.csv':
        return read_csv(str(filepath), **kwargs)
    elif suffix in ['.xlsx', '.xls']:
        return read_excel(str(filepath), **kwargs)
    elif suffix == '.json':
        data = read_json(str(filepath), **kwargs)
        return pd.DataFrame(data)
    elif suffix == '.txt':
        return read_csv(str(filepath), sep='\t', **kwargs)
    else:
        raise ValueError(f"不支持的文件格式: {suffix}")


def save_figure(fig, filepath: str, dpi: int = 300, **kwargs) -> None:
    """
    保存matplotlib图表
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        图表对象
    filepath : str
        保存路径（支持.png, .pdf, .svg, .eps等格式）
    dpi : int
        分辨率（仅对栅格格式有效）
    **kwargs : dict
        传递给savefig的其他参数
    
    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 2, 3])
    >>> save_figure(fig, 'figure.png')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
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


if __name__ == '__main__':
    # 测试代码
    print("测试文件读写工具...")
    
    # 测试CSV读写
    df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    
    write_csv(df, 'test_output.csv')
    df_read = read_csv('test_output.csv')
    print(f"\nCSV读取测试:")
    print(df_read.head())
    
    # 测试JSON读写
    data = {
        'name': 'test',
        'value': 123,
        'items': [1, 2, 3, 4, 5]
    }
    
    write_json(data, 'test_config.json')
    data_read = read_json('test_config.json')
    print(f"\nJSON读取测试:")
    print(data_read)
    
    print("\n所有测试通过！")
