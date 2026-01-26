# 美赛数学建模 - Droid CLI 模板项目

> 专为 MCM/ICM 数学建模竞赛设计的 Droid CLI 工作模板

## 快速开始

```bash
# 1. 激活虚拟环境
source venv/bin/activate

# 2. 使用 Droid CLI 开始建模
droid

# 3. 调用 MCM 建模技能
@mcm-modeling 帮我分析 [题目] 并建立模型
```

## 项目结构

```
.factory/skills/           # Droid 技能文件
├── mcm-modeling.md        # MCM 建模技能（核心）
├── batch-writer.md        # 批量写入技能
└── README.md

prompts/                   # AI Prompt 模板库
├── 01_可视化图表.md
├── 02_数据处理.md
├── 03_特征工程.md
├── 04_建模算法.md
├── 05_模型评估.md
├── 06_论文写作.md
├── 07_快速修复.md
└── 08_美赛C题数据分析算法.md

visualization/             # 可视化工具
├── publication_styles.py  # 顶会论文级样式（Nature/IEEE/NeurIPS）
├── advanced_templates/    # 高级图表模板
└── matplotlib_templates/  # 基础图表模板
```

## MCM 建模技能功能

调用 `@mcm-modeling` 技能后，Droid 会自动：

1. **文献调研** - 搜索 GitHub 优秀解决方案
2. **多算法对比** - LightGBM/XGBoost/CatBoost 等
3. **Bootstrap 置信区间** - 95% CI 估计
4. **专业特征工程** - 30+ 维度特征
5. **时间序列验证** - 防止数据泄露
6. **统计分析** - Kaplan-Meier、DID、Cohen's d
7. **顶会级可视化** - Nature/Science/IEEE 风格

## 依赖包

```bash
pip install -r requirements.txt
```

核心依赖：
- pandas, numpy, scipy
- scikit-learn, lightgbm, xgboost, catboost
- matplotlib, seaborn, SciencePlots
- lifelines (生存分析)

## 使用示例

```
# 分析数据并建模
@mcm-modeling 分析 data.csv 数据，预测 Y 变量

# 生成论文级图表
帮我用 Nature 风格绘制算法对比图

# 特征工程
对数据进行特征工程，包括时间特征、滞后特征、滚动统计
```

## 注意事项

- 每次写入代码不超过 150 行
- 使用 `encoding='utf-8-sig'` 读取中文 CSV
- 图表保存为 300 DPI PNG 格式
