# MCM C题建模指南 - AI辅助参考文档

> **用途**: 供 Claude CLI 在数学建模任务中参考，确保输出高质量代码和分析
> **适用**: MCM/ICM C题 (Data Insights 数据洞察类)

---

## 使用说明

当用户请求进行数学建模时，请按以下优先级参考本文档：
1. 根据问题类型选择合适算法 (见算法决策树)
2. 使用标准化代码模板
3. 遵循可视化规范
4. 执行质量检查清单

---

## 一、算法决策树

### 1.1 根据问题类型选择算法

```
用户问题
├── 预测未来数值/趋势
│   ├── 有时间维度 → 时间序列分析
│   │   ├── 数据量<500 → ARIMA/SARIMA
│   │   ├── 有明显季节性 → SARIMA/Prophet
│   │   ├── 数据量>1000且非线性 → LSTM
│   │   └── 需要快速实现 → Prophet
│   └── 无时间维度 → 回归分析
│       ├── 线性关系 → 线性回归/岭回归
│       ├── 非线性关系 → 随机森林/XGBoost
│       └── 高维稀疏 → Lasso/弹性网络
├── 分类/识别类别
│   ├── 二分类 → 逻辑回归/XGBoost
│   ├── 多分类 → 随机森林/XGBoost
│   └── 类别不平衡 → SMOTE + XGBoost
├── 发现数据模式/分组
│   ├── 已知分组数 → K-means
│   ├── 未知分组数 → DBSCAN/层次聚类
│   └── 需要概率输出 → GMM
├── 降维/特征分析
│   ├── 线性降维 → PCA
│   ├── 可视化目的 → t-SNE/UMAP
│   └── 分类预处理 → LDA
└── 优化/决策
    ├── 线性约束 → 线性规划 (PuLP)
    ├── 非线性优化 → scipy.optimize
    └── 不确定性分析 → 蒙特卡洛模拟
```

### 1.2 算法优先级 (O奖论文使用频率)

| 优先级 | 算法 | 使用场景 | 使用率 |
|--------|------|----------|--------|
| 1 | ARIMA/Prophet | 时间序列预测 | 85% |
| 2 | 多元回归 | 因果分析 | 80% |
| 3 | 相关性分析 | 变量关系 | 75% |
| 4 | K-means聚类 | 分组分析 | 65% |
| 5 | 随机森林/XGBoost | 复杂预测 | 60% |
| 6 | PCA | 降维 | 45% |
| 7 | LSTM | 复杂时序 | 35% |

---

## 二、标准建模流程

### 2.1 必须遵循的6步流程

```
Step 1: 数据探索 (EDA)
    ↓
Step 2: 数据预处理
    ↓
Step 3: 特征工程
    ↓
Step 4: 模型训练
    ↓
Step 5: 模型评估
    ↓
Step 6: 可视化输出
```

### 2.2 每步必须输出

| 步骤 | 必须输出 |
|------|----------|
| EDA | 数据形状、缺失值统计、描述性统计、分布图 |
| 预处理 | 处理方法说明、处理前后对比 |
| 特征工程 | 特征列表、相关性矩阵 |
| 模型训练 | 模型参数、训练过程 |
| 模型评估 | 评估指标表格、残差分析 |
| 可视化 | 至少3张专业图表 |

---

## 三、代码模板库

### 3.1 数据加载与EDA模板

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示和专业样式
plt.rcParams['font.family'] = ['Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

def load_and_explore(file_path):
    """标准数据加载与探索函数"""
    # 加载数据
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    
    print("=" * 50)
    print("1. 数据基本信息")
    print("=" * 50)
    print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"\n列名: {list(df.columns)}")
    print(f"\n数据类型:\n{df.dtypes}")
    
    print("\n" + "=" * 50)
    print("2. 缺失值分析")
    print("=" * 50)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'缺失数': missing, '缺失率%': missing_pct})
    print(missing_df[missing_df['缺失数'] > 0])
    
    print("\n" + "=" * 50)
    print("3. 描述性统计")
    print("=" * 50)
    print(df.describe())
    
    return df

# 使用示例
df = load_and_explore('data.csv')
```

### 3.2 数据预处理模板

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import KNNImputer

def preprocess_data(df, target_col, method='standard'):
    """标准数据预处理函数"""
    df = df.copy()
    
    # 1. 处理缺失值
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # 数值型: KNN填充
    if df[numeric_cols].isnull().sum().sum() > 0:
        imputer = KNNImputer(n_neighbors=5)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # 类别型: 众数填充
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    # 2. 编码类别变量
    le_dict = {}
    for col in categorical_cols:
        if col != target_col:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
    
    # 3. 分离特征和目标
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # 4. 特征缩放
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    return X_scaled, y, scaler

# 使用示例
X, y, scaler = preprocess_data(df, target_col='target')
```

### 3.3 时间序列预测模板 (ARIMA)

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def time_series_forecast(series, forecast_periods=12):
    """时间序列预测标准流程"""
    # 1. 平稳性检验
    adf_result = adfuller(series)
    print(f"ADF统计量: {adf_result[0]:.4f}")
    print(f"p值: {adf_result[1]:.4f}")
    print(f"平稳性: {'是' if adf_result[1] < 0.05 else '否，需要差分'}")
    
    # 2. 自动选择最优参数
    model = auto_arima(series, seasonal=True, m=12,
                       start_p=0, max_p=3, start_q=0, max_q=3,
                       d=None, max_d=2, trace=False,
                       error_action='ignore', suppress_warnings=True)
    
    print(f"\n最优模型: ARIMA{model.order}")
    if model.seasonal_order[0] > 0:
        print(f"季节性: SARIMA{model.seasonal_order}")
    
    # 3. 训练集/测试集划分
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    
    # 4. 模型拟合与预测
    fitted_model = ARIMA(train, order=model.order).fit()
    predictions = fitted_model.steps=len(test))
    
    # 5. 评估
    rmse = np.sqrt(mean_squared_error(test, predictions))
    mae = mean_absolute_error(test, predictions)
    mape = np.mean(np.abs((test - predictions) / test)) * 100
    
    print(f"\n模型评估:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")
    
    # 6. 未来预测
    final_model = ARIMA(series, order=model.order).fit()
    future_forecast = final_model.forecast(steps=forecast_periods)
    
    return final_model, future_forecast, {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

# 使用示例
model, forecast, metrics = time_series_forecast(df['value'], forecast_periods=12)
```

### 3.4 回归模型对比模板

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

def compare_regression_models(X, y):
    """多模型对比函数"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        
        results.append({
            'Model': name,
            'R²': r2_score(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'CV_R²_mean': cv_scores.mean(),
            'CV_R²_std': cv_scores.std()
        })
    
    results_df = pd.DataFrame(results).sort_values('R²', ascending=False)
    print("模型对比结果:")
    print(results_df.to_string(index=False))
    
    best_model_name = results_df.iloc[0]['Model']
    best_model = models[best_model_name]
    
    return results_df, best_model

# 使用示例
results, best_model = compare_regression_models(X, y)
```

### 3.5 聚类分析模板

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def clustering_analysis(X, max_k=10):
    """聚类分析标准流程"""
    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. 确定最优K值
    inertias, silhouettes = [], []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
    
    best_k = K_range[np.argmax(silhouettes)]
    print(f"最优聚类数 K = {best_k} (轮廓系数: {max(silhouettes):.4f})")
    
    # 2. 使用最优K进行聚类
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    final_labels = final_kmeans.fit_predict(X_scaled)
    
    # 3. 聚类结果统计
    cluster_stats = pd.DataFrame(X)
    cluster_stats['Cluster'] = final_labels
    summary = cluster_stats.groupby('Cluster').agg(['mean', 'count'])
    
    return final_labels, final_kmeans, {'best_k': best_k, 'silhouette': max(silhouettes)}

# 使用示例
labels, model, info = clustering_analysis(X, max_k=10)
```

---

## 四、可视化规范

### 4.1 图表样式设置 (必须使用)

```python
def set_plot_style():
    """设置专业图表样式 - 每次绑图前调用"""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 150,
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'legend.fontsize': 10,
        'lines.linewidth': 1.5
    })
    sns.set_palette("deep")

set_plot_style()
```

### 4.2 必须使用的配色方案

```python
# 主色板 (用于分类数据)
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 渐变色 (用于热力图)
CMAP_HEATMAP = 'RdBu_r'  # 相关性矩阵
CMAP_SEQUENTIAL = 'viridis'  # 连续数据

# 预测图配色
COLOR_ACTUAL = '#1f77b4'  # 实际值-蓝色
COLOR_PREDICT = '#d62728'  # 预测值-红色
COLOR_CI = '#ff7f0e'  # 置信区间-橙色
```

### 4.3 标准图表模板

#### 时间序列预测图
```python
def plot_forecast(dates, actual, predicted, ci_lower=None, ci_upper=None, title=''):
    """时间序列预测可视化"""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates[:len(actual)], actual, color='#1f77b4', label='Actual', linewidth=1.5)
    ax.plot(dates[-len(predicted):], predicted, color='#d62728', 
            linestyle='--', label='Predicted', linewidth=1.5)
    if ci_lower is not None:
        ax.fill_between(dates[-len(predicted):], ci_lower, ci_upper, 
                        alpha=0.2, color='#d62728', label='95% CI')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value')
    ax.set_title(title or 'Time Series Forecast')
    ax.legend(loc='best')
    plt.tight_layout()
    return fig
```

#### 相关性热力图
```python
def plot_correlation_matrix(df, title='Correlation Matrix'):
    """相关性矩阵热力图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, square=True, linewidths=0.5, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig
```

#### 特征重要性图
```python
def plot_feature_importance(feature_names, importances, top_n=15, title=''):
    """特征重要性条形图"""
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(idx)), importances[idx], color='#1f77b4')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel('Importance')
    ax.set_title(title or f'Top {top_n} Feature Importance')
    plt.tight_layout()
    return fig
```

#### 模型对比图
```python
def plot_model_comparison(results_df, metric='R²'):
    """模型性能对比条形图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(results_df)))
    ax.barh(results_df['Model'], results_df[metric], color=colors)
    ax.set_xlabel(metric)
    ax.set_title(f'Model Comparison by {metric}')
    for i, v in enumerate(results_df[metric]):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    plt.tight_layout()
    return fig
```

---

## 五、质量检查清单

### 5.1 代码输出前必须检查

```
□ 数据加载成功，打印了数据形状
□ 缺失值已处理并说明方法
□ 特征已标准化/归一化
□ 训练集/测试集已正确划分 (通常80/20)
□ 使用了交叉验证 (cv>=5)
□ 打印了模型评估指标表格
□ 生成了至少3张可视化图表
□ 代码包含必要注释
□ 没有硬编码的文件路径
```

### 5.2 模型评估指标要求

| 问题类型 | 必须报告的指标 |
|----------|----------------|
| 回归 | R², RMSE, MAE, MAPE |
| 分类 | Accuracy, Precision, Recall, F1, AUC |
| 聚类 | Silhouette Score, 簇内/簇间距离 |
| 时间序列 | RMSE, MAE, MAPE, 残差分析 |

### 5.3 可视化输出要求

每次建模任务必须输出以下图表:

| 图表类型 | 用途 | 必须包含 |
|----------|------|----------|
| 数据分布图 | EDA阶段 | 直方图/箱线图 |
| 相关性热力图 | 特征分析 | 数值标注 |
| 模型预测对比图 | 结果展示 | 实际vs预测 |
| 评估指标图 | 模型对比 | 多模型对比条形图 |

---

## 六、常见问题处理

### 6.1 数据问题

| 问题 | 解决方案 |
|------|----------|
| 缺失值>30% | 考虑删除该特征或使用MICE多重插补 |
| 类别不平衡 | 使用SMOTE过采样或调整class_weight |
| 异常值 | IQR方法检测，根据业务决定保留/删除 |
| 高维数据 | PCA降维或Lasso特征选择 |
| 多重共线性 | VIF检验，删除VIF>10的特征或使用岭回归 |

### 6.2 模型问题

| 问题 | 解决方案 |
|------|----------|
| 过拟合 | 增加正则化、减少特征、交叉验证 |
| 欠拟合 | 增加特征、使用更复杂模型 |
| 时序不平稳 | 差分处理、对数变换 |
| 预测偏差大 | 检查数据泄露、增加训练数据 |

---

## 七、输出格式规范

### 7.1 分析报告结构

生成分析时按以下结构组织:

```
1. 问题理解 - 任务目标、数据描述
2. 数据探索 - 基本统计、缺失值、分布
3. 数据预处理 - 处理方法、特征工程
4. 模型构建 - 算法选择理由、参数、训练
5. 结果分析 - 评估指标、可视化、发现
6. 结论建议 - 总结与改进方向
```

---

## 附录: 快速参考

### 常用导入语句

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
import xgboost as xgb
```

### 评估指标速查

```python
# 回归指标
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 分类指标
acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
```

---

**版本**: v2.0 (AI参考优化版)  
**用途**: Claude CLI 数学建模辅助参考
