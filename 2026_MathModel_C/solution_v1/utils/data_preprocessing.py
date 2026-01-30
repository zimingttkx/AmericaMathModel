# -*- coding: utf-8 -*-
"""
通用数据预处理模块
处理DWTS数据的所有预处理逻辑，包括：
1. 不同赛季的评分规则识别
2. 0分选手（已淘汰）排除
3. 评委人数归一化（3人 vs 4人）
4. N/A值处理
5. 多人淘汰周处理
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any


class SeasonRules:
    """赛季规则定义"""
    # S1-2: 排名制，TotalRank最大者淘汰
    RANKING_EARLY = (1, 2)
    # S3-27: 百分比制，TotalScore最小者淘汰
    PERCENTAGE = (3, 27)
    # S28-34: 排名制 + Bottom Two + Judges' Save
    RANKING_WITH_SAVE = (28, 34)
    
    @staticmethod
    def get_method(season: int) -> str:
        """获取赛季使用的评分方法"""
        if season <= 2:
            return 'ranking_early'
        elif season <= 27:
            return 'percentage'
        else:
            return 'ranking_with_save'
    
    @staticmethod
    def has_judges_save(season: int) -> bool:
        """是否有评委拯救环节"""
        return season >= 28


class DWTSDataProcessor:
    """DWTS数据处理器"""
    
    def __init__(self, raw_data_path: str):
        """
        初始化数据处理器
        
        Args:
            raw_data_path: 原始数据CSV路径
        """
        self.raw_df = pd.read_csv(raw_data_path)
        self.processed_data = None
        self._preprocess()
    
    def _preprocess(self):
        """预处理原始数据"""
        df = self.raw_df.copy()
        
        # 重命名列
        df = df.rename(columns={
            'celebrity_name': 'contestant',
            'ballroom_partner': 'pro_partner',
            'celebrity_industry': 'industry',
            'celebrity_homestate': 'home_state',
            'celebrity_homecountry/region': 'home_country',
            'celebrity_age_during_season': 'age'
        })
        
        # 转换为长格式（每周一行）
        self.processed_data = self._convert_to_long_format(df)
    
    def _convert_to_long_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """将宽格式数据转换为长格式"""
        records = []
        
        for _, row in df.iterrows():
            season = row['season']
            contestant = row['contestant']
            pro_partner = row['pro_partner']
            industry = row['industry']
            age = row['age']
            placement = row['placement']
            results = row['results']
            
            # 确定评委数量（S11开始有4位评委）
            has_judge4 = season >= 11
            
            for week in range(1, 12):
                # 获取各评委分数
                j1_col = f'week{week}_judge1_score'
                j2_col = f'week{week}_judge2_score'
                j3_col = f'week{week}_judge3_score'
                j4_col = f'week{week}_judge4_score'
                
                if j1_col not in df.columns:
                    continue
                
                j1 = self._parse_score(row.get(j1_col))
                j2 = self._parse_score(row.get(j2_col))
                j3 = self._parse_score(row.get(j3_col))
                j4 = self._parse_score(row.get(j4_col)) if has_judge4 else None
                
                # 跳过全0分（已淘汰）
                if j1 == 0 and j2 == 0 and j3 == 0:
                    continue
                
                # 计算评委总分和归一化分数
                if has_judge4 and j4 is not None and j4 != 0:
                    judge_scores = [j1, j2, j3, j4]
                    num_judges = 4
                else:
                    judge_scores = [j1, j2, j3]
                    num_judges = 3
                
                total_score = sum(judge_scores)
                # 归一化到30分制（3位评委每人10分）
                normalized_score = total_score * 30 / (num_judges * 10)
                
                # 判断是否被淘汰
                eliminated = self._check_elimination(results, week)
                
                records.append({
                    'season': season,
                    'week': week,
                    'contestant': contestant,
                    'pro_partner': pro_partner,
                    'industry': industry,
                    'age': age,
                    'placement': placement,
                    'judge1_score': j1,
                    'judge2_score': j2,
                    'judge3_score': j3,
                    'judge4_score': j4,
                    'num_judges': num_judges,
                    'total_judge_score': total_score,
                    'normalized_judge_score': normalized_score,
                    'eliminated': eliminated,
                    'scoring_method': SeasonRules.get_method(season),
                    'has_judges_save': SeasonRules.has_judges_save(season)
                })
        
        return pd.DataFrame(records)
    
    def _parse_score(self, value) -> float:
        """解析分数值，处理N/A"""
        if pd.isna(value) or value == 'N/A' or value == '':
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _check_elimination(self, results: str, week: int) -> bool:
        """检查选手是否在该周被淘汰"""
        if pd.isna(results):
            return False
        results = str(results).lower()
        if 'eliminated week' in results:
            try:
                elim_week = int(results.split('week')[-1].strip())
                return elim_week == week
            except:
                pass
        if 'withdrew' in results:
            return False  # 退赛不算淘汰
        return False
    
    def get_week_data(self, season: int, week: int, 
                      exclude_eliminated: bool = True) -> Optional[Dict]:
        """
        获取指定赛季和周的数据
        
        Args:
            season: 赛季号
            week: 周数
            exclude_eliminated: 是否排除已淘汰选手（评委分=0）
            
        Returns:
            包含该周所有选手数据的字典
        """
        df = self.processed_data
        week_df = df[(df['season'] == season) & (df['week'] == week)].copy()
        
        if len(week_df) == 0:
            return None
        
        if exclude_eliminated:
            # 排除评委分为0的选手（已淘汰）
            week_df = week_df[week_df['total_judge_score'] > 0]
        
        if len(week_df) == 0:
            return None
        
        # 找出本周被淘汰的选手
        eliminated_contestants = week_df[week_df['eliminated']]['contestant'].tolist()
        
        return {
            'season': season,
            'week': week,
            'contestants': week_df['contestant'].values,
            'judge_scores': week_df['normalized_judge_score'].values,
            'raw_judge_scores': week_df['total_judge_score'].values,
            'num_judges': week_df['num_judges'].values,
            'eliminated': eliminated_contestants,
            'scoring_method': week_df['scoring_method'].iloc[0],
            'has_judges_save': week_df['has_judges_save'].iloc[0],
            'n_contestants': len(week_df),
            'df': week_df
        }
    
    def get_season_data(self, season: int) -> List[Dict]:
        """获取整个赛季的数据"""
        df = self.processed_data
        weeks = sorted(df[df['season'] == season]['week'].unique())
        return [self.get_week_data(season, w) for w in weeks if self.get_week_data(season, w)]
    
    def get_contestant_features(self) -> pd.DataFrame:
        """获取选手特征数据"""
        df = self.processed_data
        features = df.groupby(['season', 'contestant']).agg({
            'industry': 'first',
            'age': 'first',
            'pro_partner': 'first',
            'placement': 'first',
            'normalized_judge_score': 'mean',
            'week': 'max'  # 存活周数
        }).reset_index()
        features = features.rename(columns={
            'normalized_judge_score': 'avg_judge_score',
            'week': 'weeks_survived'
        })
        return features


class EliminationSimulator:
    """淘汰模拟器 - 正确实现不同赛季的淘汰规则"""
    
    @staticmethod
    def simulate_ranking_early(judge_scores: np.ndarray, 
                               fan_votes: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        S1-2排名制淘汰逻辑
        规则：评委排名 + 粉丝排名 = 总排名，总排名最大者淘汰
        
        Args:
            judge_scores: 评委分数数组
            fan_votes: 粉丝投票数组（百分比或票数）
            
        Returns:
            (被淘汰选手索引, 总排名数组)
        """
        n = len(judge_scores)
        # 评委排名：分数越高排名越小（越好）
        judge_ranks = stats.rankdata(-judge_scores, method='min')
        # 粉丝排名：票数越高排名越小（越好）
        fan_ranks = stats.rankdata(-fan_votes, method='min')
        # 总排名 = 评委排名 + 粉丝排名（越小越好）
        total_ranks = judge_ranks + fan_ranks
        # 总排名最大者被淘汰
        eliminated_idx = np.argmax(total_ranks)
        return eliminated_idx, total_ranks
    
    @staticmethod
    def simulate_percentage(judge_pct: np.ndarray, 
                           fan_pct: np.ndarray) -> Tuple[int, np.ndarray]:
        """
        S3-27百分比制淘汰逻辑
        规则：评委百分比 + 粉丝百分比 = 总分，总分最小者淘汰
        
        Args:
            judge_pct: 评委百分比数组
            fan_pct: 粉丝百分比数组
            
        Returns:
            (被淘汰选手索引, 总分数组)
        """
        total_scores = judge_pct + fan_pct
        # 总分最小者被淘汰
        eliminated_idx = np.argmin(total_scores)
        return eliminated_idx, total_scores
    
    @staticmethod
    def simulate_ranking_with_save(judge_scores: np.ndarray,
                                   fan_votes: np.ndarray,
                                   raw_judge_scores: np.ndarray = None) -> Tuple[int, np.ndarray, Dict]:
        """
        S28+排名制 + Bottom Two + Judges' Save淘汰逻辑
        规则：
        1. 评委排名 + 粉丝排名 = 总排名
        2. 总排名最差的两人进入Bottom Two
        3. 评委投票决定谁被淘汰（通常救评委分较高者）
        
        Args:
            judge_scores: 归一化评委分数
            fan_votes: 粉丝投票
            raw_judge_scores: 原始评委分数（用于Judges' Save决策）
            
        Returns:
            (被淘汰选手索引, 总排名数组, 详细信息字典)
        """
        if raw_judge_scores is None:
            raw_judge_scores = judge_scores
            
        n = len(judge_scores)
        
        # 计算排名
        judge_ranks = stats.rankdata(-judge_scores, method='min')
        fan_ranks = stats.rankdata(-fan_votes, method='min')
        total_ranks = judge_ranks + fan_ranks
        
        # 找出Bottom Two（总排名最大的两人）
        sorted_indices = np.argsort(total_ranks)[::-1]  # 从大到小
        bottom_two = sorted_indices[:2]
        
        # Judges' Save: 评委分较高者被救回
        if raw_judge_scores[bottom_two[0]] > raw_judge_scores[bottom_two[1]]:
            eliminated_idx = bottom_two[1]
            saved_idx = bottom_two[0]
        else:
            eliminated_idx = bottom_two[0]
            saved_idx = bottom_two[1]
        
        details = {
            'bottom_two': bottom_two.tolist(),
            'saved_idx': saved_idx,
            'eliminated_idx': eliminated_idx,
            'bottom_two_judge_scores': raw_judge_scores[bottom_two].tolist()
        }
        
        return eliminated_idx, total_ranks, details
    
    @classmethod
    def simulate_elimination(cls, week_data: Dict, 
                            fan_votes: np.ndarray) -> Tuple[int, Any, Dict]:
        """
        根据赛季规则模拟淘汰
        
        Args:
            week_data: 周数据字典
            fan_votes: 粉丝投票数组
            
        Returns:
            (被淘汰选手索引, 排名/分数数组, 详细信息)
        """
        method = week_data['scoring_method']
        judge_scores = week_data['judge_scores']
        
        if method == 'ranking_early':
            elim_idx, ranks = cls.simulate_ranking_early(judge_scores, fan_votes)
            return elim_idx, ranks, {'method': 'ranking_early'}
        
        elif method == 'percentage':
            # 转换为百分比
            judge_pct = judge_scores / judge_scores.sum() * 100
            fan_pct = fan_votes / fan_votes.sum() * 100
            elim_idx, scores = cls.simulate_percentage(judge_pct, fan_pct)
            return elim_idx, scores, {'method': 'percentage'}
        
        else:  # ranking_with_save
            raw_scores = week_data.get('raw_judge_scores', judge_scores)
            elim_idx, ranks, details = cls.simulate_ranking_with_save(
                judge_scores, fan_votes, raw_scores
            )
            details['method'] = 'ranking_with_save'
            return elim_idx, ranks, details


class FanVoteEstimator:
    """粉丝投票估算器基类"""
    
    @staticmethod
    def scores_to_percentages(scores: np.ndarray) -> np.ndarray:
        """将分数转换为百分比"""
        total = scores.sum()
        if total == 0:
            return np.ones(len(scores)) / len(scores) * 100
        return scores / total * 100
    
    @staticmethod
    def validate_fan_votes(fan_votes: np.ndarray, 
                          tolerance: float = 0.01) -> bool:
        """验证粉丝投票是否有效（和为100%）"""
        return abs(fan_votes.sum() - 100) < tolerance


def load_and_process_data(data_path: str) -> DWTSDataProcessor:
    """便捷函数：加载并处理数据"""
    return DWTSDataProcessor(data_path)


if __name__ == '__main__':
    # 测试代码
    processor = DWTSDataProcessor('../../data/2026_MCM_Problem_C_Data.csv')
    
    # 测试不同赛季
    for season in [1, 5, 15, 28, 30]:
        week_data = processor.get_week_data(season, 3)
        if week_data:
            print(f"\nSeason {season}, Week 3:")
            print(f"  Method: {week_data['scoring_method']}")
            print(f"  Has Judges Save: {week_data['has_judges_save']}")
            print(f"  Contestants: {len(week_data['contestants'])}")
            print(f"  Eliminated: {week_data['eliminated']}")
