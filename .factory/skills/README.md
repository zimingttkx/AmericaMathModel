# Skills 使用指南

本目录包含项目专用的 Droid Skills，用于增强 AI 助手在特定任务上的能力。

## 文件结构

```
.factory/skills/
├── mcm-modeling.md          # ✨ MCM/ICM 数学建模竞赛专用 Skill
├── batch-writer.md          # Batch Writer Skill（大文件分批写入）
├── batch_writer_tool.py     # Python 工具脚本
└── README.md               # 本文件
```

---

## 🏆 MCM Modeling Skill（数学建模竞赛）

### 功能概述
专为美国大学生数学建模竞赛（MCM/ICM）设计的开发助手，提供：
- 数据分析与预处理规范
- 机器学习/深度学习建模指南
- 论文级可视化图表生成
- 模型评估与验证流程
- 代码规范与最佳实践

### 使用方法

在对话中直接描述你的需求，Droid 会自动参考此 skill：

```
"帮我读取数据并进行探索性分析"
"使用 XGBoost 进行分类，并绘制特征重要性图"
"绘制算法性能对比图，论文级质量"
"对比多种回归模型的性能"
```

### 详细参考
更详细的 Prompt 模板请查看 `prompts/` 目录下的文件。

---

## 📝 Batch Writer Skill（大文件分批写入）

### 快速开始

强制执行150行写入限制的 skill，自动将大文件拆分成多个批次写入。

## 如何使用

### 方法 1: 自动模式（推荐）

在创建或编辑文件时，告诉 Droid 使用 batch-writer skill：

```
使用 batch-writer skill 创建一个包含500行代码的文件 large_file.py
```

Droid 会自动：
1. 检测内容行数
2. 拆分成多个批次（每批最多150行）
3. 在逻辑边界处分割（不破坏函数/类定义）
4. 顺序写入每个批次
5. 验证完整性

### 方法 2: 手动使用工具

在代码中导入并使用工具：

```python
from .factory.skills.batch_writer_tool import BatchWriter, check_write_size

# 检查内容大小
content = generate_large_content()
exceeds_limit, line_count = check_write_size(content)

if exceeds_limit:
    print(f"内容有 {line_count} 行，需要分批写入")
    
    # 查看批次计划
    print(BatchWriter.format_batch_summary(content))
    
    # 获取分割后的批次
    batches = BatchWriter.split_content(content)
    for i, batch in enumerate(batches, 1):
        print(f"批次 {i}: {len(batch.split(chr(10)))} 行")
```

## 示例输出

当创建一个500行的文件时，您会看到：

```
📊 Content Analysis:
   Total lines: 500
   Batches needed: 4

📦 Batch Breakdown:
   Batch 1: Lines 1-150 (150 lines)
   Batch 2: Lines 151-300 (150 lines)
   Batch 3: Lines 301-450 (150 lines)
   Batch 4: Lines 451-500 (50 lines)

✅ File created: large_file.py
✅ All 4 batches written successfully
✅ Integrity verified: 500 lines written
```

## 核心特性

✅ **智能分割**：在逻辑边界（函数/类结束、空行）处分割，不破坏代码结构  
✅ **严格限制**：每个批次最多150行，绝不会超过  
✅ **完整性验证**：写入后自动验证所有内容是否正确  
✅ **详细反馈**：提供清晰的批次信息和进度反馈  

## 技术细节

- **最大行数**: 150 行/批次
- **最小行数**: 50 行/批次（避免过多小批次）
- **分割策略**: 优先在空行、块结尾处分割
- **验证方法**: 逐行对比原始内容和写入内容

## 测试

运行测试验证工具是否正常工作：

```bash
python3 .factory/skills/batch_writer_tool.py
```

预期输出应显示批次分解信息和完整性验证通过。
