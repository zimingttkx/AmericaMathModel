"""Seaborn 统计图表模板 - 数学建模常用"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def correlation_heatmap(df, figsize=(10, 8), cmap='RdBu_r', annot=True):
    """相关性热力图"""
    fig, ax = plt.subplots(figsize=figsize)
    corr = df.select_dtypes(include=[np.number]).corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap=cmap, annot=annot, fmt='.2f',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, ax


def distribution_grid(df, cols=None, figsize=(12, 10)):
    """多变量分布图网格"""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns[:9]
    n = len(cols)
    nrows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, 3, figsize=figsize)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color='steelblue')
        axes[i].set_title(col, fontsize=10)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    return fig, axes


def boxplot_comparison(df, x, y, figsize=(10, 6), palette='Set2'):
    """分组箱线图比较"""
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=df, x=x, y=y, palette=palette, ax=ax)
    ax.set_title(f'{y} by {x}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax


def pairplot_analysis(df, hue=None, vars=None, figsize=(12, 12)):
    """成对关系图"""
    if vars is None:
        vars = df.select_dtypes(include=[np.number]).columns[:5]
    g = sns.pairplot(df, hue=hue, vars=vars, diag_kind='kde',
                     plot_kws={'alpha': 0.6}, height=2.5)
    g.fig.suptitle('Pairwise Relationships', y=1.02, fontsize=14)
    return g


def violin_plot(df, x, y, figsize=(10, 6), palette='muted'):
    """小提琴图"""
    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(data=df, x=x, y=y, palette=palette, ax=ax)
    ax.set_title(f'Distribution of {y} by {x}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig, ax


def regression_plot(df, x, y, figsize=(10, 6)):
    """回归散点图"""
    fig, ax = plt.subplots(figsize=figsize)
    sns.regplot(data=df, x=x, y=y, scatter_kws={'alpha': 0.5}, ax=ax)
    ax.set_title(f'{y} vs {x} with Regression Line', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig, ax


def categorical_count(df, x, hue=None, figsize=(10, 6), palette='Set2'):
    """分类计数图"""
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(data=df, x=x, hue=hue, palette=palette, ax=ax)
    ax.set_title(f'Count of {x}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.legend(title=hue) if hue else None
    plt.tight_layout()
    return fig, ax


if __name__ == '__main__':
    # 示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100) * 2,
        'C': np.random.randn(100) + 1,
        'Category': np.random.choice(['X', 'Y', 'Z'], 100)
    })
    
    fig, ax = correlation_heatmap(df[['A', 'B', 'C']])
    plt.savefig('seaborn_demo.png', dpi=150)
    print("Demo saved to seaborn_demo.png")
