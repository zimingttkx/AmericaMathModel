"""
字体配置模块
提供字体设置和管理功能
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings


class FontConfig:
    """字体配置类"""
    
    # 推荐的英文字体
    ENGLISH_FONTS = {
        'times': 'Times New Roman',
        'arial': 'Arial',
        'helvetica': 'Helvetica',
        'calibri': 'Calibri',
        'georgia': 'Georgia',
    }
    
    # 推荐的中文字体
    CHINESE_FONTS = {
        'simhei': 'SimHei',           # 黑体
        'simsun': 'SimSun',           # 宋体
        'kaiti': 'KaiTi',             # 楷体
        'fangsong': 'FangSong',       # 仿宋
        'microsoft': 'Microsoft YaHei', # 微软雅黑
    }
    
    @staticmethod
    def list_available_fonts():
        """列出系统中所有可用的字体"""
        fonts = sorted([f.name for f in fm.fontManager.ttflist])
        return fonts
    
    @staticmethod
    def check_font_available(font_name):
        """
        检查字体是否可用
        
        Parameters:
        -----------
        font_name : str
            字体名称
        
        Returns:
        --------
        bool : 字体是否可用
        """
        available_fonts = FontConfig.list_available_fonts()
        return font_name in available_fonts
    
    @staticmethod
    def set_english_font(font='times'):
        """
        设置英文字体
        
        Parameters:
        -----------
        font : str
            字体名称，可选: 'times', 'arial', 'helvetica', 'calibri', 'georgia'
            或直接传入字体名称字符串
        """
        if font in FontConfig.ENGLISH_FONTS:
            font_name = FontConfig.ENGLISH_FONTS[font]
        else:
            font_name = font
        
        if not FontConfig.check_font_available(font_name):
            warnings.warn(f"Font '{font_name}' not available. Using default font.")
            return
        
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = [font_name]
        print(f"English font set to: {font_name}")
    
    @staticmethod
    def set_chinese_font(font='simhei'):
        """
        设置中文字体
        
        Parameters:
        -----------
        font : str
            字体名称，可选: 'simhei', 'simsun', 'kaiti', 'fangsong', 'microsoft'
            或直接传入字体名称字符串
        """
        if font in FontConfig.CHINESE_FONTS:
            font_name = FontConfig.CHINESE_FONTS[font]
        else:
            font_name = font
        
        if not FontConfig.check_font_available(font_name):
            warnings.warn(f"Font '{font_name}' not available. Using default font.")
            # 尝试备选字体
            for backup_font in ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']:
                if FontConfig.check_font_available(backup_font):
                    font_name = backup_font
                    warnings.warn(f"Using backup font: {backup_font}")
                    break
        
        plt.rcParams['font.sans-serif'] = [font_name]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        print(f"Chinese font set to: {font_name}")
    
    @staticmethod
    def set_mixed_font(english='times', chinese='simhei'):
        """
        设置中英文混合字体
        
        Parameters:
        -----------
        english : str
            英文字体
        chinese : str
            中文字体
        """
        FontConfig.set_english_font(english)
        FontConfig.set_chinese_font(chinese)
        print("Mixed font configuration applied.")
    
    @staticmethod
    def set_font_sizes(small=9, medium=10, large=12):
        """
        设置字体大小
        
        Parameters:
        -----------
        small : int
            小字体大小（用于刻度标签）
        medium : int
            中等字体大小（用于坐标轴标签）
        large : int
            大字体大小（用于标题）
        """
        plt.rcParams['font.size'] = medium
        plt.rcParams['axes.labelsize'] = medium
        plt.rcParams['axes.titlesize'] = large
        plt.rcParams['xtick.labelsize'] = small
        plt.rcParams['ytick.labelsize'] = small
        plt.rcParams['legend.fontsize'] = small
        print(f"Font sizes set: small={small}, medium={medium}, large={large}")
    
    @staticmethod
    def reset_fonts():
        """重置为默认字体设置"""
        plt.rcParams.update(plt.rcParamsDefault)
        print("Font settings reset to default.")


# 创建全局实例
font_config = FontConfig()


if __name__ == '__main__':
    # 测试代码
    import numpy as np
    
    # 列出可用字体（前20个）
    print("Available fonts (first 20):")
    fonts = font_config.list_available_fonts()
    for i, font in enumerate(fonts[:20]):
        print(f"  {i+1}. {font}")
    
    # 检查特定字体
    print("\nChecking fonts:")
    print(f"  Times New Roman: {font_config.check_font_available('Times New Roman')}")
    print(f"  Arial: {font_config.check_font_available('Arial')}")
    print(f"  SimHei: {font_config.check_font_available('SimHei')}")
    
    # 设置字体并绘图
    font_config.set_english_font('times')
    font_config.set_font_sizes(small=9, medium=11, large=13)
    
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    plt.figure(figsize=(8, 5))
    plt.plot(x, y)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Font Configuration Test')
    plt.grid(True, alpha=0.3)
    plt.savefig('test_font_config.png', dpi=300, bbox_inches='tight')
    plt.show()
