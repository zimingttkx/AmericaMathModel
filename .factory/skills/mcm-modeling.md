# MCM/ICM 数学建模竞赛开发 Skill

**Description**: 美国大学生数学建模竞赛（MCM/ICM）专用开发助手，提供数据分析、建模、可视化、论文写作的完整支持。

## 适用场景

当用户进行以下任务时自动激活：
- 数学建模竞赛代码开发
- 数据分析与预处理
- 机器学习/深度学习建模
- 科学可视化图表生成
- 论文图表和结果展示

## 核心原则

### 1. 代码规范
- 使用 Python 3.x，优先使用 pandas、numpy、sklearn、matplotlib、seaborn
- 代码必须包含清晰的中文注释
- 变量命名规范，函数职责单一
- 所有图表必须达到论文发表级别质量

### 2. 输出标准
- 图表分辨率：300 DPI（论文用）或 600 DPI（高质量）
- 图表格式：PNG（默认）或 PDF（矢量图）
- 保存路径：`workspace/figures/` 或 `visualization/outputs/`
- 文件命名：`{类型}_{描述}_{日期}.png`

---

## 数据处理模块

### 数据读取与检查
```python
# 标准数据读取流程
import pandas as pd
import numpy as np

def load_and_check(file_path):
    """读取数据并进行基础检查"""
    df = pd.read_csv(file_path)
    print(f"数据形状: {df.shape}")
    print(f"列名: {df.columns.tolist()}")
    print(f"缺失值:\n{df.isnull().sum()}")
    print(f"数据类型:\n{df.dtypes}")
    return df
```

### 缺失值处理策略
| 缺失比例 | 处理方法 |
|---------|---------|
| < 5% | 删除或均值/中位数填充 |
| 5-30% | KNN填充或多重插补 |
| > 30% | 考虑删除该列或建立缺失指示变量 |

### 异常值检测
- **3σ原则**: 适用于正态分布数据
- **IQR方法**: Q1-1.5*IQR ~ Q3+1.5*IQR
- **Isolation Forest**: 高维数据异常检测

---

## 可视化模块

### ⭐ 顶会论文级可视化要求

> **重要**: 图表必须达到 Nature/Science/IEEE/NeurIPS 等顶级期刊/会议的出版标准。

#### 推荐可视化库

| 库 | 用途 | 安装 |
|---|------|------|
| **SciencePlots** | 期刊风格样式 (IEEE/Nature) | `pip install SciencePlots` |
| **Plotnine** | ggplot2 语法 (R风格) | `pip install plotnine` |
| **ProPlot** | 增强版 matplotlib | `pip install proplot` |
| **Seaborn** | 统计可视化 | `pip install seaborn` |

#### 使用 SciencePlots (推荐)
```python
import scienceplots
import matplotlib.pyplot as plt

# Nature 期刊风格
plt.style.use(['science', 'nature'])

# IEEE 论文风格 (黑白打印友好)
plt.style.use(['science', 'ieee'])

# 高对比度 + 网格
plt.style.use(['science', 'bright', 'grid'])

# 中文支持
plt.style.use(['science', 'no-latex', 'cjk-sc-font'])
```

#### 期刊尺寸规范

| 期刊 | 单栏宽度 | 双栏宽度 | 字体 |
|-----|---------|---------|------|
| **Nature** | 89mm (3.5") | 183mm (7.2") | Arial/Helvetica, 5-7pt |
| **Science** | 2.3" | 4.6" | Arial/Helvetica |
| **IEEE** | 3.5" | 7.16" | Times New Roman, 8-10pt |
| **NeurIPS/ICML** | 5.5" | - | Times, 9-10pt |

#### 专业配色方案

```python
# Nature 配色 (清晰、专业)
NATURE_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

# 色盲友好配色 (Paul Tol's Bright)
COLORBLIND_SAFE = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE']

# IEEE 黑白友好
IEEE_COLORS = ['#000000', '#E69F00', '#56B4E9', '#009E73', '#F0E442']
```

#### 图表规范检查清单

- [ ] **字体**: Sans-serif (Nature/Science) 或 Serif (IEEE)
- [ ] **字号**: 最小 5pt，推荐 7-10pt
- [ ] **分辨率**: ≥300 DPI (打印), ≥600 DPI (IEEE)
- [ ] **坐标轴**: 标签完整，含单位 (如 "Time (s)")
- [ ] **图例**: 位置合适，不遮挡数据
- [ ] **配色**: 色盲友好，打印友好
- [ ] **无装饰**: 无背景网格线，无3D效果
- [ ] **面板标签**: 多图使用 (a), (b), (c) 标注

#### 本项目可视化模块

```python
# 使用项目内置的出版级样式
from visualization.publication_styles import (
    apply_nature_style,    # Nature 期刊风格
    apply_ieee_style,      # IEEE 论文风格
    apply_science_style,   # Science 期刊风格
    apply_neurips_style,   # NeurIPS/ICML 顶会风格
    create_figure,         # 创建符合规范的图形
    add_panel_labels,      # 添加面板标签
    save_publication_figure,  # 保存多格式图形
    plot_with_error_band,  # 误差带曲线
    plot_comparison_bars,  # 算法对比柱状图
    plot_heatmap_publication,  # 出版级热力图
)

# 示例
apply_nature_style()
fig, ax = create_figure(style='nature', width='single')
ax.plot(x, y)
save_publication_figure(fig, 'figure1', formats=['pdf', 'png'])
```

---

### 图表风格配置
```python
import matplotlib.pyplot as plt
import seaborn as sns

# 论文级图表配置
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'Arial',
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# 中文支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
```

### 常用配色方案
```python
# 学术配色（推荐）
ACADEMIC_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 高对比度（竞赛用）
HIGH_CONTRAST = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']

# Nature风格
NATURE_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
```

### 图表类型选择指南
| 数据类型 | 推荐图表 | 用途 |
|---------|---------|------|
| 时间序列 | 折线图 | 趋势展示 |
| 分类对比 | 柱状图/分组柱状图 | 算法性能对比 |
| 相关性 | 热力图 | 特征相关性分析 |
| 分布 | 直方图/箱线图 | 数据分布展示 |
| 预测评估 | 散点图+对角线 | 预测vs实际 |
| 聚类结果 | 2D/3D散点图 | 聚类可视化 |

---

## 建模算法模块

### ⭐ 专业建模要求（Outstanding Paper 标准）

> **重要**: 参考 MCM 优秀论文，建模必须满足以下专业要求：

#### 0. 文献调研（首要步骤）
**在开始建模前，必须先进行文献调研**：

1. **搜索优秀论文**：
   - GitHub 搜索：`MCM [年份] [题号] outstanding` 或 `美赛 [题号] O奖`
   - 知网/Google Scholar：搜索相关领域的学术论文
   - arXiv：搜索最新的方法论文

2. **分析优秀方案的方法**：
   - 记录使用的算法和模型
   - 记录特征工程方法
   - 记录评估指标和验证方法
   - 记录创新点和亮点

3. **将优秀方法加入对比**：
   ```python
   # 示例：发现优秀论文使用了 Grey Model + LightGBM + Cox
   # 则必须在算法对比中包含这些方法
   models_from_papers = {
       'Grey_GM11': GreyModel(),      # 来自论文 A
       'LightGBM': lgb.LGBMRegressor(), # 来自论文 B  
       'Cox_PH': CoxPHFitter(),        # 来自论文 C
   }
   ```

4. **文献调研模板**：
   ```markdown
   ## 文献调研记录
   
   ### 搜索关键词
   - GitHub: "MCM 2025 C outstanding"
   - Scholar: "Olympic medal prediction machine learning"
   
   ### 优秀论文方法汇总
   | 来源 | 方法 | 创新点 | 是否采用 |
   |-----|------|-------|---------|
   | Paper A | LightGBM + SHAP | 可解释性分析 | ✅ |
   | Paper B | Cox 生存分析 | 首次获奖预测 | ✅ |
   | Paper C | DID 双重差分 | 因果推断 | ✅ |
   | Paper D | Grey Model | 小样本预测 | ✅ |
   ```

#### 1. 多算法对比（必须）
每个建模任务必须实现 **至少 5 种算法** 进行对比：

| 类别 | 必选算法 | 说明 |
|-----|---------|------|
| 线性模型 | Ridge, Lasso, ElasticNet | 基线模型，可解释性强 |
| 传统集成 | RandomForest, GradientBoosting | 稳健的集成方法 |
| **高级 GBDT** | **LightGBM, XGBoost, CatBoost** | ⭐ 竞赛必备，性能最优 |
| 神经网络 | MLP (MLPRegressor/Classifier) | 捕捉复杂非线性 |
| 支持向量机 | SVR / SVC | 小样本效果好 |

```python
# 高级 GBDT 安装（macOS 需要 libomp）
# brew install libomp
# pip install lightgbm xgboost catboost

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor, CatBoostClassifier
```

#### 2. Bootstrap 置信区间（必须）
所有预测结果必须提供 **95% 置信区间**：

```python
def bootstrap_prediction(models, X, n_bootstrap=100):
    """Bootstrap 置信区间估计"""
    predictions = []
    for _ in range(n_bootstrap):
        # 随机选择模型子集
        selected = np.random.choice(len(models), size=len(models), replace=True)
        preds = np.mean([models[i].predict(X) for i in selected], axis=0)
        predictions.append(preds)
    
    predictions = np.array(predictions)
    mean_pred = np.mean(predictions, axis=0)
    ci_lower = np.percentile(predictions, 2.5, axis=0)
    ci_upper = np.percentile(predictions, 97.5, axis=0)
    return mean_pred, ci_lower, ci_upper
```

#### 3. 特征工程（30+ 维度）
专业建模需要丰富的特征工程：

| 特征类别 | 示例 | 数量要求 |
|---------|------|---------|
| 基础统计 | mean, std, min, max, sum | 5+ |
| 时间特征 | lag, rolling_mean, trend | 5+ |
| 比率特征 | ratio, percentage, growth_rate | 5+ |
| 交互特征 | feature_A * feature_B | 5+ |
| 聚合特征 | group_mean, group_std | 5+ |
| 多样性指标 | HHI, entropy, gini | 3+ |

#### 4. 时间序列验证（必须）
使用 **TimeSeriesSplit** 而非随机划分：

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

#### 5. 专业统计分析方法

##### Kaplan-Meier 生存分析
用于分析"首次发生某事件的时间"：
```python
from lifelines import KaplanMeierFitter, CoxPHFitter

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed=events)
kmf.plot_survival_function()

# Cox 比例风险模型（需要 penalizer 避免奇异矩阵）
cph = CoxPHFitter(penalizer=0.1)
cph.fit(data, duration_col='T', event_col='E')
```

##### DID 双重差分法
用于因果推断（如教练效应）：
```python
import statsmodels.api as sm

# DID 回归: Y = β0 + β1*Treated + β2*Post + β3*Treated*Post + ε
# β3 即为 DID 效应
X = data[['Treated', 'Post', 'Treated_Post']]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
did_effect = model.params['Treated_Post']
```

##### 效应量分析
```python
from scipy import stats

# Cohen's d 效应量
def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (group1.mean() - group2.mean()) / pooled_std

# 效应量解释: |d| < 0.2 小, 0.2-0.8 中, > 0.8 大
```

---

### 算法选择指南

#### 分类问题
| 场景 | 推荐算法 | 优先级 |
|-----|---------|-------|
| 基线模型 | Logistic Regression | 1 |
| 中等复杂度 | Random Forest | 2 |
| **高性能需求** | **LightGBM / XGBoost / CatBoost** | ⭐ 3 |
| 非线性边界 | SVM (RBF kernel) | 4 |
| 深度学习 | MLP | 5 |

#### 回归问题
| 场景 | 推荐算法 | 优先级 |
|-----|---------|-------|
| 线性关系 | Ridge / Lasso / ElasticNet | 1 |
| 非线性关系 | Random Forest Regressor | 2 |
| **高性能需求** | **LightGBM / XGBoost / CatBoost** | ⭐ 3 |
| 神经网络 | MLPRegressor | 4 |

#### 时间序列
| 场景 | 推荐算法 | 优先级 |
|-----|---------|-------|
| 短期预测 | ARIMA | 1 |
| 季节性数据 | SARIMA / Prophet | 2 |
| **机器学习方法** | **LightGBM + 时间特征** | ⭐ 3 |
| 复杂模式 | LSTM | 4 |

#### 聚类问题
| 场景 | 推荐算法 | 优先级 |
|-----|---------|-------|
| 已知K值 | K-Means | 1 |
| 未知K值 | 层次聚类 + 轮廓系数 | 2 |
| 密度聚类 | DBSCAN | 3 |

#### 生存分析
| 场景 | 推荐算法 | 优先级 |
|-----|---------|-------|
| 生存曲线 | Kaplan-Meier | 1 |
| 风险因素 | Cox 比例风险模型 | 2 |
| 参数模型 | Weibull / Exponential | 3 |

#### 因果推断
| 场景 | 推荐算法 | 优先级 |
|-----|---------|-------|
| 政策效应 | DID 双重差分 | 1 |
| 匹配方法 | PSM 倾向得分匹配 | 2 |
| 工具变量 | 2SLS | 3 |

### 标准建模流程
```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# 1. 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. 模型训练
model.fit(X_train_scaled, y_train)

# 4. 交叉验证
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# 5. 测试集评估
y_pred = model.predict(X_test_scaled)
```

---

## 模型评估模块

### ⭐ 专业评估要求

#### 多维度评估（必须）
每个模型必须报告以下指标：

| 指标 | 回归 | 分类 | 说明 |
|-----|------|------|------|
| R² | ✅ | - | 解释方差比例 |
| RMSE | ✅ | - | 均方根误差 |
| MAE | ✅ | - | 平均绝对误差 |
| MAPE | ✅ | - | 平均绝对百分比误差 |
| Accuracy | - | ✅ | 准确率 |
| Precision | - | ✅ | 精确率 |
| Recall | - | ✅ | 召回率 |
| F1-Score | - | ✅ | F1 分数 |
| AUC-ROC | - | ✅ | ROC 曲线下面积 |

#### 模型对比可视化（必须）
```python
# 雷达图对比
def plot_model_radar(results_df, metrics, save_path):
    """多模型多指标雷达图对比"""
    from math import pi
    
    categories = metrics
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for idx, row in results_df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'])
        ax.fill(angles, values, alpha=0.1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

# 热力图对比
def plot_model_heatmap(results_df, save_path):
    """模型性能热力图"""
    metrics = ['R2', 'RMSE', 'MAE', 'MAPE']
    data = results_df.set_index('Model')[metrics]
    
    # 归一化到 0-1
    data_norm = (data - data.min()) / (data.max() - data.min())
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_norm, annot=data.round(4), cmap='RdYlGn', 
                fmt='', linewidths=0.5)
    plt.title('Model Performance Comparison')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

### 分类评估指标
```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# 完整评估
def evaluate_classification(y_true, y_pred, y_proba=None):
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    if y_proba is not None:
        print(f"AUC:       {roc_auc_score(y_true, y_proba):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
```

### 回归评估指标
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"MAPE: {mape:.2f}%")
```

---

## 特征工程模块

### 特征选择方法
1. **方差阈值**: 删除低方差特征
2. **相关性过滤**: 删除高相关特征（>0.9）
3. **模型重要性**: 基于 RandomForest/XGBoost 特征重要性
4. **RFE**: 递归特征消除

### 特征变换
- **标准化**: StandardScaler（均值0，方差1）
- **归一化**: MinMaxScaler（范围[0,1]）
- **对数变换**: 处理右偏分布
- **多项式特征**: 捕捉非线性关系

### 降维方法
- **PCA**: 线性降维，保留95%方差
- **t-SNE**: 非线性降维，用于可视化
- **LDA**: 监督降维，最大化类间距离

---

## 论文图表规范

### 图表标题格式
```
Figure X: [描述性标题]
Table X: [描述性标题]
```

### 图表要素检查清单
- [ ] 标题清晰、描述准确
- [ ] 坐标轴标签完整（含单位）
- [ ] 图例位置合适、不遮挡数据
- [ ] 字体大小适中（≥10pt）
- [ ] 配色协调、对比度足够
- [ ] 分辨率≥300 DPI

### 常用图表代码模板

#### 算法性能对比图
```python
def plot_algorithm_comparison(algorithms, datasets, scores, save_path):
    """绘制算法性能对比分组柱状图"""
    x = np.arange(len(datasets))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (algo, score) in enumerate(zip(algorithms, scores)):
        ax.bar(x + i*width, score, width, label=algo)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Algorithm Performance Comparison')
    ax.set_xticks(x + width)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

#### 预测结果散点图
```python
def plot_prediction_scatter(y_true, y_pred, save_path):
    """绘制预测值vs实际值散点图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolors='none')
    
    # 对角线
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', lw=2, label='Ideal')
    
    # R²标注
    r2 = r2_score(y_true, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes, fontsize=12)
    
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Prediction Performance')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
```

---

## 项目文件结构

```
AmericaMathModel/
├── data/                    # 原始数据
│   └── raw/
├── workspace/               # 工作目录
│   ├── figures/            # 生成的图表
│   ├── model_code/         # 模型代码
│   ├── results/            # 结果输出
│   └── problem_analysis/   # 问题分析
├── visualization/           # 可视化模块
│   └── outputs/            # 可视化输出
├── prompts/                 # Prompt模板参考
└── requirements.txt         # 依赖包
```

---

## 快速命令参考

### 数据探索
```
"帮我读取 data.csv 并进行数据探索分析"
"检查数据的缺失值和异常值"
"绘制特征相关性热力图"
```

### 建模
```
"使用随机森林进行分类，并进行5折交叉验证"
"对比 XGBoost、LightGBM、RandomForest 的性能"
"使用 ARIMA 预测未来12个月的数据"
```

### 可视化
```
"绘制算法性能对比柱状图，保存为论文级质量"
"绘制预测结果散点图，包含R²值"
"绘制聚类结果的2D可视化"
```

### 评估
```
"评估分类模型，输出完整的评估报告"
"绘制ROC曲线和混淆矩阵"
"进行学习曲线分析，诊断过拟合"
```

---

## 注意事项

1. **数据泄露**: 先划分数据集，再进行特征工程
2. **随机种子**: 所有随机操作设置 `random_state=42`
3. **模型保存**: 使用 joblib 保存训练好的模型
4. **版本控制**: 重要结果及时 git commit
5. **代码复用**: 将常用函数封装到 utils 模块

---

## 依赖包要求

### 核心依赖（必须安装）
```bash
# 基础数据科学
pip install pandas numpy scipy matplotlib seaborn

# 机器学习
pip install scikit-learn

# 高级 GBDT（竞赛必备）
brew install libomp  # macOS 需要
pip install lightgbm xgboost catboost

# 生存分析
pip install lifelines

# 统计建模
pip install statsmodels

# Jupyter
pip install jupyter nbformat nbconvert

# ⭐ 顶会论文级可视化（推荐）
pip install SciencePlots  # 需要 LaTeX
pip install plotnine      # ggplot2 语法
pip install proplot       # 增强版 matplotlib
```

### requirements.txt 示例
```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
catboost>=1.2.0
lifelines>=0.27.0
statsmodels>=0.14.0
jupyter>=1.0.0
SciencePlots>=2.0.0
plotnine>=0.12.0
```

---

## 参考资料

详细的 Prompt 模板请参考 `prompts/` 目录：
- `01_可视化图表.md` - 图表绘制详细模板
- `02_数据处理.md` - 数据处理详细模板
- `03_特征工程.md` - 特征工程详细模板
- `04_建模算法.md` - 建模算法详细模板
- `05_模型评估.md` - 模型评估详细模板
- `06_论文写作.md` - 论文写作详细模板
- `07_快速修复.md` - 代码调试详细模板
- `08_美赛C题数据分析算法.md` - C题专用算法模板
