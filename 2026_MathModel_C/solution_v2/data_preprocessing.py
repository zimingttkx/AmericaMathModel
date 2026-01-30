"""
Q1 数据预处理模块
- 加载原始数据
- 处理缺失值(N/A)
- 构建每周有效参赛者集合 A_t
- 计算评委总分/平均分
- 提取淘汰标签 E_t
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 数据路径
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
RAW_DATA_PATH = DATA_DIR / "2026_MCM_Problem_C_Data.csv"


def load_raw_data() -> pd.DataFrame:
    """加载原始数据"""
    df = pd.read_csv(RAW_DATA_PATH, encoding='utf-8-sig')
    print(f"数据形状: {df.shape[0]} 行 × {df.shape[1]} 列")
    print(f"赛季范围: {df['season'].min()} - {df['season'].max()}")
    return df


def parse_elimination_week(results: str) -> Optional[int]:
    """
    从 results 字段解析淘汰周次
    返回: 淘汰周次 (int) 或 None (未被淘汰/冠军等)
    """
    if pd.isna(results):
        return None
    results = str(results).strip()
    
    # 匹配 "Eliminated Week X" 格式
    match = re.search(r'Eliminated Week (\d+)', results, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # 特殊情况: Withdrew (退赛)
    if 'Withdrew' in results or 'Quit' in results:
        return -1  # 用 -1 表示退赛
    
    # 1st/2nd/3rd Place 等 - 未被淘汰
    return None


def get_placement(results: str, placement_col: int) -> int:
    """获取最终排名"""
    if pd.notna(placement_col):
        return int(placement_col)
    return 999  # 未知排名


def extract_judge_scores(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    提取每个赛季的评委评分数据
    返回: {season: DataFrame} 字典
    """
    # 找出所有评分列
    score_cols = [col for col in df.columns if re.match(r'week\d+_judge\d+_score', col)]
    
    # 解析周次和评委编号
    def parse_score_col(col: str) -> Tuple[int, int]:
        match = re.match(r'week(\d+)_judge(\d+)_score', col)
        return int(match.group(1)), int(match.group(2))
    
    season_data = {}
    for season in df['season'].unique():
        season_df = df[df['season'] == season].copy()
        season_data[season] = season_df
    
    return season_data


def compute_weekly_scores(df: pd.DataFrame, max_weeks: int = 11) -> pd.DataFrame:
    """
    计算每位选手每周的评委总分和平均分
    
    返回 DataFrame 包含:
    - celebrity_name, season
    - week{t}_total_score: 第t周评委总分
    - week{t}_avg_score: 第t周评委平均分
    - week{t}_judge_count: 第t周有效评委数
    - elimination_week: 淘汰周次
    """
    result_rows = []
    
    for idx, row in df.iterrows():
        record = {
            'celebrity_name': row['celebrity_name'],
            'ballroom_partner': row['ballroom_partner'],
            'season': row['season'],
            'placement': row['placement'],
            'results': row['results'],
            'elimination_week': parse_elimination_week(row['results']),
            'celebrity_industry': row.get('celebrity_industry', ''),
        }
        
        for week in range(1, max_weeks + 1):
            scores = []
            for judge in range(1, 5):  # 最多4位评委
                col = f'week{week}_judge{judge}_score'
                if col in row.index:
                    val = row[col]
                    # 处理 N/A 和有效分数
                    if pd.notna(val) and val != 'N/A' and val != 0:
                        try:
                            score = float(val)
                            if score > 0:  # 排除0分(已淘汰)
                                scores.append(score)
                        except (ValueError, TypeError):
                            pass
            
            if scores:
                record[f'week{week}_total_score'] = sum(scores)
                record[f'week{week}_avg_score'] = np.mean(scores)
                record[f'week{week}_judge_count'] = len(scores)
            else:
                record[f'week{week}_total_score'] = np.nan
                record[f'week{week}_avg_score'] = np.nan
                record[f'week{week}_judge_count'] = 0
        
        result_rows.append(record)
    
    return pd.DataFrame(result_rows)


def build_weekly_active_sets(processed_df: pd.DataFrame) -> Dict[int, Dict[int, List[str]]]:
    """
    构建每个赛季每周的有效参赛者集合 A_{s,t}
    
    返回: {season: {week: [celebrity_names]}}
    """
    active_sets = {}
    
    for season in processed_df['season'].unique():
        season_df = processed_df[processed_df['season'] == season]
        active_sets[season] = {}
        
        # 找出该赛季最大周数
        max_week = 1
        for week in range(1, 12):
            col = f'week{week}_total_score'
            if col in season_df.columns and season_df[col].notna().any():
                max_week = week
        
        for week in range(1, max_week + 1):
            col = f'week{week}_total_score'
            # 有效参赛者: 该周有评分 (非NaN)
            active = season_df[season_df[col].notna()]['celebrity_name'].tolist()
            active_sets[season][week] = active
    
    return active_sets


def build_elimination_sets(processed_df: pd.DataFrame) -> Dict[int, Dict[int, List[str]]]:
    """
    构建每个赛季每周的淘汰者集合 E_{s,t}
    
    返回: {season: {week: [eliminated_celebrity_names]}}
    """
    elimination_sets = {}
    
    for season in processed_df['season'].unique():
        season_df = processed_df[processed_df['season'] == season]
        elimination_sets[season] = {}
        
        for week in range(1, 12):
            # 该周被淘汰的选手
            eliminated = season_df[season_df['elimination_week'] == week]['celebrity_name'].tolist()
            if eliminated:
                elimination_sets[season][week] = eliminated
    
    return elimination_sets


def get_season_summary(processed_df: pd.DataFrame) -> pd.DataFrame:
    """
    生成赛季汇总信息
    """
    summary = []
    for season in sorted(processed_df['season'].unique()):
        season_df = processed_df[processed_df['season'] == season]
        
        # 计算有效周数
        max_week = 0
        for week in range(1, 12):
            col = f'week{week}_total_score'
            if col in season_df.columns and season_df[col].notna().any():
                max_week = week
        
        summary.append({
            'season': season,
            'num_contestants': len(season_df),
            'max_weeks': max_week,
            'winner': season_df[season_df['placement'] == 1]['celebrity_name'].values[0] if (season_df['placement'] == 1).any() else 'N/A',
        })
    
    return pd.DataFrame(summary)


def preprocess_all() -> Tuple[pd.DataFrame, Dict, Dict, pd.DataFrame]:
    """
    执行完整预处理流程
    
    返回:
    - processed_df: 处理后的选手数据
    - active_sets: 每周有效参赛者集合
    - elimination_sets: 每周淘汰者集合
    - season_summary: 赛季汇总
    """
    print("=" * 50)
    print("开始数据预处理...")
    print("=" * 50)
    
    # 1. 加载原始数据
    raw_df = load_raw_data()
    
    # 2. 计算每周评分
    print("\n计算每周评委评分...")
    processed_df = compute_weekly_scores(raw_df)
    print(f"处理后数据形状: {processed_df.shape}")
    
    # 3. 构建有效参赛者集合
    print("\n构建每周有效参赛者集合 A_{s,t}...")
    active_sets = build_weekly_active_sets(processed_df)
    
    # 4. 构建淘汰者集合
    print("构建每周淘汰者集合 E_{s,t}...")
    elimination_sets = build_elimination_sets(processed_df)
    
    # 5. 生成赛季汇总
    print("\n生成赛季汇总...")
    season_summary = get_season_summary(processed_df)
    
    print("\n" + "=" * 50)
    print("数据预处理完成!")
    print("=" * 50)
    
    return processed_df, active_sets, elimination_sets, season_summary


# 测试
if __name__ == "__main__":
    processed_df, active_sets, elimination_sets, season_summary = preprocess_all()
    
    print("\n赛季汇总:")
    print(season_summary.to_string(index=False))
    
    # 示例: 查看第3赛季数据
    print("\n\n示例 - 第3赛季:")
    print(f"参赛者: {active_sets[3][1]}")
    print(f"第1周淘汰: {elimination_sets[3].get(1, [])}")
    print(f"第2周淘汰: {elimination_sets[3].get(2, [])}")
    
    # 查看某选手数据
    s3_df = processed_df[processed_df['season'] == 3]
    print("\n第3赛季选手评分:")
    cols = ['celebrity_name', 'elimination_week', 'week1_total_score', 'week2_total_score', 'week3_total_score']
    print(s3_df[cols].to_string(index=False))
