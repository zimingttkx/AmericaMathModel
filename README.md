# 美国数学建模竞赛 - 建模工具库 🏆

> 专为建模选手打造的算法与可视化工具库

## 🎯 快速导航

| 需求 | 目录 | 说明 |
|------|------|------|
| 🧮 **需要算法？** | [`algorithms/`](algorithms/) | 查看算法分类和实现 |
| 📊 **需要画图？** | [`visualization/`](visualization/) | 查看图表模板 |
| 💡 **需要示例？** | [`examples/`](examples/) | 查看完整案例 |
| 🔧 **需要工具？** | [`utils/`](utils/) | 数据处理和验证工具 |
| 📚 **需要参考？** | [`reference/`](reference/) | 算法速查和比赛技巧 |
| 💼 **比赛工作？** | [`workspace/`](workspace/) | 在此编写代码 |

## 📚 算法速查表

| 问题类型 | 推荐算法 | 位置 |
|---------|---------|------|
| 🎯 **路径优化** | 遗传算法、模拟退火、粒子群 | [`algorithms/optimization/`](algorithms/optimization/) |
| 📈 **数据预测** | 时间序列、回归分析、神经网络 | [`algorithms/prediction/`](algorithms/prediction/) |
| ⭐ **方案评价** | AHP、TOPSIS、模糊评价 | [`algorithms/evaluation/`](algorithms/evaluation/) |
| 🕸️ **网络问题** | 最短路径、网络流、图着色 | [`algorithms/graph_network/`](algorithms/graph_network/) |
| 📊 **统计分析** | 假设检验、相关分析、聚类 | [`algorithms/statistics/`](algorithms/statistics/) |

## 🎨 可视化速查表

| 图表类型 | 使用场景 | 模板位置 |
|---------|---------|---------|
| 📈 **折线图** | 趋势变化、时间序列 | [`visualization/matplotlib_templates/line_charts.py`](visualization/matplotlib_templates/line_charts.py) |
| 📊 **柱状图** | 数值比较、分类统计 | [`visualization/matplotlib_templates/bar_charts.py`](visualization/matplotlib_templates/bar_charts.py) |
| 🔥 **热力图** | 相关性矩阵、密度分布 | [`visualization/matplotlib_templates/heatmaps.py`](visualization/matplotlib_templates/heatmaps.py) |
| 🌐 **3D图形** | 三维数据展示 | [`visualization/matplotlib_templates/3d_plots.py`](visualization/matplotlib_templates/3d_plots.py) |
| 🕸️ **网络图** | 关系展示、图论可视化 | [`visualization/networkx_templates/`](visualization/networkx_templates/) |

## 🚀 快速开始

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 或使用国内镜像加速
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 使用示例

```python
# 示例1: 使用遗传算法求解优化问题
from algorithms.optimization.genetic_algorithm import GeneticAlgorithm

ga = GeneticAlgorithm(pop_size=100, generations=200)
best_solution = ga.optimize(objective_function)

# 示例2: 绘制专业图表
from visualization.matplotlib_templates import line_charts
from visualization.style_config import paper_style

paper_style.apply()  # 应用论文风格
line_charts.plot_multi_lines(data, title="Results", save_path="figure1.png")

# 示例3: 数据处理
from utils.data_processing import normalize, remove_outliers

clean_data = remove_outliers(raw_data)
normalized_data = normalize(clean_data)
```

### 3. 比赛工作流程

```
1. 问题分析 → workspace/problem_analysis/
   ├── 记录问题理解
   ├── 确定建模思路
   └── 选择合适算法

2. 编写代码 → workspace/model_code/
   ├── 参考 examples/ 中的示例
   ├── 使用 algorithms/ 中的算法
   └── 调用 utils/ 中的工具

3. 生成图表 → workspace/figures/
   ├── 使用 visualization/ 模板
   ├── 确保高分辨率 (300 DPI)
   └── 统一配色和字体

4. 输出结果 → workspace/results/
   ├── 保存计算结果
   ├── 导出数据表格
   └── 整理模型输出
```

## 📖 目录结构

```
AmericaMathModel/
├── algorithms/              # 建模算法库
│   ├── optimization/        # 优化算法
│   ├── prediction/          # 预测模型
│   ├── evaluation/          # 评价模型
│   ├── graph_network/       # 图论与网络
│   └── statistics/          # 统计分析
├── visualization/           # 可视化图表库
│   ├── matplotlib_templates/  # Matplotlib 模板
│   ├── seaborn_templates/     # Seaborn 模板
│   ├── plotly_templates/      # Plotly 交互式图表
│   ├── networkx_templates/    # 网络图可视化
│   └── style_config/          # 样式配置
├── utils/                   # 实用工具
├── examples/                # 完整示例
├── reference/               # 参考资料
├── data/                    # 数据文件夹
└── workspace/               # 工作区（比赛时使用）
```

## 💡 使用技巧

### 算法选择建议

**优化问题：**
- 连续变量 → 梯度下降、牛顿法
- 离散变量 → 遗传算法、模拟退火
- 有约束条件 → 线性规划、整数规划
- 多目标优化 → 粒子群算法、NSGA-II

**预测问题：**
- 时间序列数据 → ARIMA、Prophet、LSTM
- 线性关系 → 线性回归、岭回归
- 非线性关系 → 随机森林、神经网络
- 分类问题 → 逻辑回归、SVM、决策树

**评价问题：**
- 主观权重 → AHP（层次分析法）
- 客观权重 → 熵权法、主成分分析
- 综合评价 → TOPSIS、灰色关联分析

### 可视化建议

**美赛图表要求：**
1. ✅ 高分辨率（至少 300 DPI）
2. ✅ 专业配色（避免过于鲜艳）
3. ✅ 清晰标注（标题、坐标轴、图例）
4. ✅ 统一字体（Times New Roman 或 Arial）
5. ✅ 适当留白（不要过于拥挤）

**图表选择：**
- 展示趋势 → 折线图
- 比较数值 → 柱状图、雷达图
- 展示分布 → 直方图、箱线图、小提琴图
- 展示关系 → 散点图、热力图
- 展示比例 → 饼图、堆叠图
- 展示网络 → 网络图、桑基图

## 🎓 学习资源

- **算法详解**: 查看 [`algorithms/README.md`](algorithms/README.md)
- **可视化指南**: 查看 [`visualization/README.md`](visualization/README.md)
- **完整示例**: 查看 [`examples/README.md`](examples/README.md)
- **比赛技巧**: 查看 [`reference/competition_tips.md`](reference/competition_tips.md)

## 📦 项目特点

✨ **专注建模**: 只包含建模人员需要的代码和工具  
🎯 **结构清晰**: 按功能分类，快速定位所需内容  
📚 **即用即查**: 每个模块都有说明和示例  
⚡ **轻量高效**: 不包含大量论文和教材，体积小  
🔧 **可扩展**: 可以随时添加自己的算法和模板  

## 🤝 贡献指南

比赛后可以将自己的优秀代码整理到对应模块：

1. 算法实现 → `algorithms/` 对应分类
2. 可视化模板 → `visualization/` 对应分类
3. 完整案例 → `examples/`
4. 实用工具 → `utils/`

## 📝 更新日志

- **2026-01-25**: 初始化项目结构
  - ✅ 创建完整目录结构
  - ✅ 配置 Python 环境
  - ✅ 编写核心文档

## 📧 联系方式

如有问题或建议，欢迎交流！

---

**祝您在美赛中取得优异成绩！🏆**

*Good luck and have fun modeling!*
