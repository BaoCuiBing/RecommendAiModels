"""
欢迎页面模块
显示系统简介和功能导航
"""

import os
import tkinter as tk
from tkinter import ttk
import logging
from typing import Dict, Any
from datetime import datetime

# 导入基础页面类
from ui.pages.base_page import BasePage

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("welcome_page")


class WelcomePage(BasePage):
    """欢迎页面类"""
    
    def __init__(self, parent, controller=None, **kwargs):
        """
        初始化欢迎页面
        
        Args:
            parent: 父级窗口或Frame
            controller: 页面控制器
            **kwargs: 传递给BasePage的参数
        """
        super().__init__(parent, controller, **kwargs)
        logger.debug("欢迎页面已初始化")
    
    def _create_widgets(self):
        """创建页面小部件"""
        # 设置页面标题
        self.title_label = ttk.Label(
            self, 
            text="欢迎使用智能题目推荐系统", 
            font=("Helvetica", 18, "bold")
        )
        self.title_label.pack(pady=(20, 10))
        
        # 添加版本信息
        version = self.get_shared_data('version', '1.0.0')
        last_update = self.get_shared_data('last_update', datetime.now().strftime('%Y-%m-%d'))
        
        version_frame = ttk.Frame(self)
        version_frame.pack(fill=tk.X, padx=20, pady=5)
        
        ttk.Label(
            version_frame, 
            text=f"版本: {version} | 最后更新: {last_update}",
            font=("Helvetica", 10)
        ).pack(side=tk.RIGHT)
        
        # 创建内容区域
        content_frame = ttk.Frame(self)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
        
        # 左侧：系统介绍
        intro_frame = ttk.LabelFrame(content_frame, text="系统简介")
        intro_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)
        
        intro_text = tk.Text(intro_frame, wrap=tk.WORD, width=40, height=15, padx=10, pady=10)
        intro_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加系统介绍文本
        intro_content = """
        智能题目推荐系统是一个基于Python的教育辅助工具，利用机器学习算法为用户推荐最合适的练习题目。

        系统特点：
        
        1. 基于scikit-learn和Joblib的推荐算法
        2. 支持生成随机测试数据进行模型训练
        3. 集成Matplotlib和Seaborn进行数据可视化
        4. 友好的Tkinter图形界面
        5. 支持导入自定义数据集增量学习
        6. 多线程优化，提高性能
        
        本系统适用于教育机构、在线学习平台或个人学习者，可根据用户的历史作答记录，智能推荐适合的学习内容。
        """
        
        intro_text.insert("1.0", intro_content)
        intro_text.config(state=tk.DISABLED)  # 设置为只读
        
        # 右侧：功能导航
        nav_frame = ttk.LabelFrame(content_frame, text="功能导航")
        nav_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        
        # 添加功能按钮
        self._create_nav_button(
            nav_frame, 
            "数据生成", 
            "生成训练和测试数据",
            lambda: self.controller.show_page("DataPage")
        )
        
        self._create_nav_button(
            nav_frame, 
            "模型训练", 
            "训练推荐算法模型",
            lambda: self.controller.show_page("TrainPage")
        )
        
        self._create_nav_button(
            nav_frame, 
            "性能监控", 
            "查看模型训练性能",
            lambda: self.controller.show_page("MonitorPage")
        )
        
        self._create_nav_button(
            nav_frame, 
            "题目推荐", 
            "获取个性化题目推荐",
            lambda: self.controller.show_page("RecommendPage")
        )
        
        self._create_nav_button(
            nav_frame, 
            "系统设置", 
            "配置系统参数",
            lambda: self.controller.show_page("SettingsPage")
        )
        
        # 添加底部信息
        footer_frame = ttk.Frame(self)
        footer_frame.pack(fill=tk.X, padx=20, pady=10)
        
        ttk.Label(
            footer_frame, 
            text="Copyright © 2023 智能题目推荐系统团队",
            font=("Helvetica", 8)
        ).pack(side=tk.LEFT)
    
    def _create_nav_button(self, parent, title, description, command):
        """
        创建导航按钮
        
        Args:
            parent: 父级窗口或Frame
            title: 按钮标题
            description: 功能描述
            command: 点击回调函数
        """
        # 创建按钮框架
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=10, pady=8)
        
        # 创建按钮
        button = ttk.Button(
            button_frame, 
            text=title,
            command=command,
            width=15
        )
        button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 添加描述
        ttk.Label(
            button_frame, 
            text=description
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _show_help(self):
        """显示帮助信息"""
        if self.controller:
            self.controller.show_help()
    
    def refresh(self):
        """刷新页面内容"""
        # 如果需要动态更新内容，可以在这里实现
        pass
    
    def on_show(self):
        """页面显示时的回调"""
        logger.debug("欢迎页面显示")
        
        # 更新版本信息
        version = self.get_shared_data('version', '1.0.0')
        last_update = self.get_shared_data('last_update', datetime.now().strftime('%Y-%m-%d'))
        
        # 可以在这里添加页面显示时的其他逻辑


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("欢迎页面测试")
    root.geometry("800x600")
    
    # 模拟控制器
    class MockController:
        def __init__(self):
            self.shared_data = {
                'version': '1.0.0',
                'last_update': '2023-10-01'
            }
        
        def show_page(self, page_name):
            print(f"切换到页面: {page_name}")
        
        def show_help(self):
            print("显示帮助")
    
    # 创建欢迎页面
    page = WelcomePage(root, MockController())
    
    # 显示窗口
    root.mainloop() 