# MCM/ICM 数学建模工具箱 🏆

<p align="center">
  <strong>专为美国大学生数学建模竞赛 (MCM/ICM) 设计的一站式解决方案</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/MCM-2025-orange.svg" alt="MCM">
</p>

---

## ✨ 特性亮点

| 模块 | 功能 | 说明 |
|------|------|------|
| 📊 **可视化** | 顶刊级图表 | Nature / Science / IEEE 风格模板 |
| 🤖 **建模算法** | 机器学习 | LightGBM / XGBoost / CatBoost / 神经网络 |
| 📈 **统计分析** | 专业统计 | 假设检验 / 回归分析 / 生存分析 |
| 🔧 **特征工程** | 自动化 | 30+ 维度特征自动生成 |
| 📝 **论文写作** | Prompt 模板 | 8 大类建模场景覆盖 |

## 🚀 快速开始

```bash
# 克隆项目
git clone https://github.com/zimingttkx/AmericaMathModel.git
cd AmericaMathModel

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 📁 项目结构

```
AmericaMathModel/
├── prompts/                    # 📝 AI Prompt 模板库
│   ├── 01_可视化图表.md        # 数据可视化指南
│   ├── 02_数据处理.md          # 数据清洗与预处理
│   ├── 03_特征工程.md          # 特征构建方法
│   ├── 04_建模算法.md          # 算法选择与实现
│   ├── 05_模型评估.md          # 模型验证与评估
│   ├── 06_论文写作.md          # 论文撰写技巧
│   ├── 07_快速修复.md          # 常见问题解决
│   └── 08_美赛C题数据分析算法.md
│
├── visualization/              # 📊 可视化工具库
│   ├── publication_styles.py   # 顶刊论文样式 (Nature/IEEE/NeurIPS)
│   ├── advanced_templates/     # 高级图表模板
│   ├── matplotlib_templates/   # Matplotlib 模板
│   ├── seaborn_templates/      # Seaborn 统计图表
│   ├── plotly_templates/       # Plotly 交互式图表
│   └── networkx_templates/     # 网络图可视化
│
└── .factory/skills/            # 🤖 Droid AI 技能
    ├── mcm-modeling.md         # MCM 建模技能
    └── batch-writer.md         # 批量写入技能
```

## 🎯 核心功能

### 1. Prompt 模板库

针对数学建模全流程的 8 大 Prompt 模板：

```
数据处理 → 特征工程 → 建模算法 → 模型评估 → 可视化 → 论文写作
```

### 2. 顶刊级可视化

```python
from visualization.publication_styles import apply_nature_style

# 一键应用 Nature 风格
apply_nature_style()
plt.plot(x, y)
plt.savefig('figure.png', dpi=300)
```

支持风格：
- 🔬 **Nature** - 自然科学顶刊
- 🔌 **IEEE** - 工程技术期刊
- 🧠 **NeurIPS** - 机器学习顶会

### 3. 统计分析工具

- 假设检验 (t-test, ANOVA, Chi-square)
- 回归分析 (线性/非线性/逻辑回归)
- 时间序列 (ARIMA, Prophet)
- 生存分析 (Kaplan-Meier)

## 📦 依赖环境

| 类别 | 包名 |
|------|------|
| 科学计算 | numpy, scipy, pandas |
| 可视化 | matplotlib, seaborn, plotly |
| 机器学习 | scikit-learn |
| 优化算法 | pulp, cvxpy |
| 统计分析 | statsmodels, pingouin |
| 图论网络 | networkx |
| 时间序列 | prophet |

## 📖 使用示例

### 数据分析流程

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# 1. 读取数据
df = pd.read_csv('data.csv', encoding='utf-8-sig')

# 2. 特征工程
X = df.drop('target', axis=1)
y = df['target']

# 3. 模型训练
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 4. 模型评估
score = model.score(X_test, y_test)
print(f'R² Score: {score:.4f}')
```

## 🏅 适用赛题

- **MCM Problem A** - 连续型建模
- **MCM Problem B** - 离散型建模  
- **MCM Problem C** - 数据洞察
- **ICM Problem D** - 运筹/网络优化
- **ICM Problem E** - 环境科学
- **ICM Problem F** - 政策建模

## 📄 License

MIT License - 自由使用，助力美赛！

---

<p align="center">
  <strong>⭐ 如果这个项目对你有帮助，请给一个 Star！</strong>
</p>
