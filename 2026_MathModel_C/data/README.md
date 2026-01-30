# DWTS 数据集说明文档

> 本文档记录 2026 MCM Problem C 所使用的所有数据文件的来源、结构和字段说明。
> 
> **重要提示**: 所有数据均为原始数据 (Raw Data)，未经任何预处理或清洗。

---

## 1. 官方题目数据

### 1.1 `2026_MCM_Problem_C_Data.csv`

| 属性 | 值 |
|------|-----|
| **来源** | COMAP 官方题目数据 |
| **下载地址** | https://www.immchallenge.org/mcm/2026_MCM_Problem_C_Data.csv |
| **数据类型** | CSV |
| **记录数** | 421 行 (选手) |
| **季数范围** | Season 1 - Season 34 |
| **列数** | 53 列 |

#### 字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `celebrity_name` | string | 名人参赛者姓名 | Jerry Rice, Mark Cuban |
| `ballroom_partner` | string | 职业舞伴姓名 | Cheryl Burke, Derek Hough |
| `celebrity_industry` | string | 明星职业类别 | Athlete, Actor/Actress, Singer/Rapper |
| `celebrity_homestate` | string | 明星家乡州 (美国) | Ohio, Maine, California |
| `celebrity_homecountry/region` | string | 明星家乡国家/地区 | United States, England, New Zealand |
| `celebrity_age_during_season` | int | 明星在该季的年龄 | 32, 29, 45 |
| `season` | int | 节目季数 | 1, 2, 3, ..., 34 |
| `results` | string | 赛季成绩描述 | 1st Place, Eliminated Week 3 |
| `placement` | int | 赛季最终排名 (1=冠军) | 1, 2, 3, ... |
| `weekX_judgeY_score` | float | 第X周评委Y的评分 (1-10分) | 7, 8.5, 9 |

#### 评分字段说明

- 格式: `week{1-11}_judge{1-4}_score`
- 评分范围: 1分 (最低) 至 10分 (最高)
- 特殊值:
  - `N/A`: 该周未安排第四位评委，或节目未播出
  - `0`: 选手已被淘汰

#### 数据注意事项

1. 部分周次分数包含小数 (如 8.5)，因每位明星表演多支舞蹈，最终分数为平均值
2. 评委按评分舞蹈顺序排列，"评委Y"可能并非每周/每季都由同一人担任
3. 每季参赛明星人数和播出周数不尽相同
4. 第15季为全明星赛季，参赛者均为往季回归选手
5. 某些周次可能无人淘汰，另一些周次可能淘汰多人

---

## 2. 补充数据 (GitHub: howisonlab/dwts_dataset)

> **来源**: https://github.com/howisonlab/dwts_dataset
> 
> **数据说明**: 从 Wikipedia 表格爬取的 DWTS 数据集，用于数据工程教学目的。
> 
> **许可证**: GPL-3.0

### 2.1 `celebrities.csv`

| 属性 | 值 |
|------|-----|
| **记录数** | 368 行 |
| **数据类型** | CSV |

#### 字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `celebrity_id` | int | 名人唯一标识符 | 1, 2, 3 |
| `celebrity` | string | 名人姓名 | Erin Andrews, Evan Lysacek |
| `notability` | string | 名人知名度来源/职业 | ESPN sportscaster, Olympic figure skater |

---

### 2.2 `professionals.csv`

| 属性 | 值 |
|------|-----|
| **记录数** | 55 行 |
| **数据类型** | CSV |

#### 字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `professional_id` | int | 职业舞者唯一标识符 | 1, 2, 3 |
| `professional` | string | 职业舞者姓名 | Maksim Chmerkovskiy, Anna Trebunskaya |

---

### 2.3 `couples.csv`

| 属性 | 值 |
|------|-----|
| **记录数** | 414 行 |
| **数据类型** | CSV |

#### 字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `couple_id` | int | 组合唯一标识符 | 1, 2, 3 |
| `couple` | string | 组合名称 (名人 & 舞者) | Erin & Maks, Evan & Anna |

---

### 2.4 `judges.csv`

| 属性 | 值 |
|------|-----|
| **记录数** | 39 行 |
| **数据类型** | CSV |

#### 字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `judge_id` | int | 评委唯一标识符 | 1, 2, 3 |
| `judge` | string | 评委姓名 | Carrie Ann Inaba, Len Goodman, Bruno Tonioli |

---

### 2.5 `performances.csv`

| 属性 | 值 |
|------|-----|
| **记录数** | 2986 行 |
| **数据类型** | CSV |

#### 字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `performance_id` | int | 表演唯一标识符 | 820, 907, 908 |
| `season` | int | 季数 | 10, 11, 12 |
| `week` | string | 周次 | 4, 5, Finals |
| `week_theme_id` | int | 周主题ID (外键) | 1, 2, 3 |
| `couple_id` | int | 组合ID (外键) | 1, 2, 3 |
| `professional_id` | int | 职业舞者ID (外键) | 1, 2, 3 |
| `primary_dance_style_id` | int | 主要舞蹈风格ID (外键) | 1, 2, 3 |

---

### 2.6 `scores.csv`

| 属性 | 值 |
|------|-----|
| **记录数** | 9734 行 |
| **数据类型** | CSV |

#### 字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `score_id` | int | 评分唯一标识符 | 1, 2, 3 |
| `performance_id` | int | 表演ID (外键) | 820, 907 |
| `judge_id` | int | 评委ID (外键) | 1, 2, 3 |
| `judge_score` | float | 评委评分 (满分10) | 6.5, 9.0, 8.0 |

---

### 2.7 `week_themes.csv`

| 属性 | 值 |
|------|-----|
| **记录数** | 105 行 |
| **数据类型** | CSV |

#### 字段说明

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `week_theme_id` | int | 周主题唯一标识符 | 1, 2, 3 |
| `week_theme` | string | 周主题名称 | Double-score Week, Acoustic Week, Disney Night |

---

## 3. 数据关系图 (ER Diagram)

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   celebrities   │     │   professionals │     │     judges      │
├─────────────────┤     ├─────────────────┤     ├─────────────────┤
│ celebrity_id PK │     │ professional_id │     │ judge_id PK     │
│ celebrity       │     │ PK              │     │ judge           │
│ notability      │     │ professional    │     └────────┬────────┘
└────────┬────────┘     └────────┬────────┘              │
         │                       │                       │
         └───────────┬───────────┘                       │
                     │                                   │
              ┌──────▼──────┐                           │
              │   couples   │                           │
              ├─────────────┤                           │
              │ couple_id PK│                           │
              │ couple      │                           │
              └──────┬──────┘                           │
                     │                                   │
              ┌──────▼──────────────┐                   │
              │    performances     │                   │
              ├─────────────────────┤                   │
              │ performance_id PK   │                   │
              │ season              │                   │
              │ week                │                   │
              │ week_theme_id FK    │──┐               │
              │ couple_id FK        │  │               │
              │ professional_id FK  │  │               │
              │ primary_dance_      │  │               │
              │ style_id FK         │  │               │
              └──────┬──────────────┘  │               │
                     │                  │               │
              ┌──────▼──────┐    ┌─────▼─────┐        │
              │   scores    │    │week_themes│        │
              ├─────────────┤    ├───────────┤        │
              │ score_id PK │    │week_theme_│        │
              │ performance_│    │id PK      │        │
              │ id FK       │    │week_theme │        │
              │ judge_id FK │────┴───────────┘        │
              │ judge_score │◄────────────────────────┘
              └─────────────┘
```

---

## 4. 数据下载时间

| 文件 | 下载时间 (UTC+8) |
|------|------------------|
| `2026_MCM_Problem_C_Data.csv` | 2026-01-30 08:09 |
| `celebrities.csv` | 2026-01-30 08:08 |
| `couples.csv` | 2026-01-30 08:08 |
| `judges.csv` | 2026-01-30 08:08 |
| `performances.csv` | 2026-01-30 08:08 |
| `professionals.csv` | 2026-01-30 08:08 |
| `scores.csv` | 2026-01-30 08:08 |
| `week_themes.csv` | 2026-01-30 08:08 |

---

## 5. 参考资料

1. **COMAP MCM 2026**: https://www.immchallenge.org/mcm/index.html
2. **GitHub dwts_dataset**: https://github.com/howisonlab/dwts_dataset
3. **Wikipedia DWTS Competitors**: https://en.wikipedia.org/wiki/List_of_Dancing_with_the_Stars_(American_TV_series)_competitors
