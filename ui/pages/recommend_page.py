"""
题目推荐页面模块
提供用户选择、题目推荐和结果展示功能
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import json
import threading
import time
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 导入基础页面类
from ui.pages.base_page import BasePage

# 导入线程池工具
from utils.thread_pool import submit_task, get_task_status

# 导入配置工具
from utils.config import get_config, get_models_dir

# 导入用户模型
from user_model import UserModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("recommend_page")

# 从train_page导入中文字体配置
from ui.pages.train_page import configure_matplotlib_chinese

# 配置中文字体
configure_matplotlib_chinese()

class RecommendPage(BasePage):
    """题目推荐页面类"""
    
    def __init__(self, parent, controller=None, **kwargs):
        """
        初始化题目推荐页面
        
        Args:
            parent: 父级窗口或Frame
            controller: 页面控制器
            **kwargs: 传递给BasePage的参数
        """
        # 推荐相关变量
        self.is_recommending = False
        self.recommending_thread = None
        self.model_path = tk.StringVar(value="")
        self.user_id = tk.StringVar(value="")
        self.difficulty_level = tk.IntVar(value=3)  # 1-5
        self.num_recommendations = tk.IntVar(value=5)
        self.recommended_items = []
        self.last_recommendations = {}
        self.status_var = tk.StringVar(value="就绪")  # 添加状态变量
        
        super().__init__(parent, controller, **kwargs)
        logger.debug("题目推荐页面已初始化")
    
    def _create_widgets(self):
        """创建页面小部件"""
        # 设置页面标题
        self.title_label = ttk.Label(
            self, 
            text="题目推荐", 
            font=("Helvetica", 18, "bold")
        )
        self.title_label.pack(pady=(20, 10))
        
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 左侧：用户和模型选择
        left_frame = ttk.LabelFrame(main_frame, text="推荐配置", width=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10), pady=10)
        left_frame.pack_propagate(False)  # 防止子组件影响frame大小
        
        # 模型选择
        model_frame = ttk.Frame(left_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(model_frame, text="推荐模型:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(model_frame, textvariable=self.model_path, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(model_frame, text="浏览...", command=self._browse_model).pack(side=tk.LEFT, padx=(5, 0))
        
        # 用户ID输入
        user_frame = ttk.Frame(left_frame)
        user_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(user_frame, text="用户ID:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(user_frame, textvariable=self.user_id, width=15).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(user_frame, text="历史记录", command=self._show_history).pack(side=tk.LEFT, padx=(5, 0))
        
        # 难度级别选择
        difficulty_frame = ttk.LabelFrame(left_frame, text="难度级别")
        difficulty_frame.pack(fill=tk.X, padx=10, pady=10)
        
        difficulty_scale = ttk.Scale(
            difficulty_frame, 
            from_=1, 
            to=5, 
            orient=tk.HORIZONTAL, 
            variable=self.difficulty_level,
            length=200
        )
        difficulty_scale.pack(fill=tk.X, padx=10, pady=5)
        
        # 难度级别标签
        level_frame = ttk.Frame(difficulty_frame)
        level_frame.pack(fill=tk.X, padx=10, pady=0)
        
        for i in range(1, 6):
            ttk.Label(level_frame, text=str(i)).pack(side=tk.LEFT, expand=True)
        
        # 难度级别描述
        level_desc = ["入门", "基础", "中等", "进阶", "挑战"]
        
        desc_frame = ttk.Frame(difficulty_frame)
        desc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        for desc in level_desc:
            ttk.Label(desc_frame, text=desc).pack(side=tk.LEFT, expand=True)
        
        # 推荐数量选择
        num_frame = ttk.Frame(left_frame)
        num_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(num_frame, text="推荐数量:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            num_frame, 
            from_=1, 
            to=20, 
            increment=1, 
            textvariable=self.num_recommendations,
            width=5
        ).pack(side=tk.LEFT)
        
        # 用户特征
        features_frame = ttk.LabelFrame(left_frame, text="用户特征")
        features_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 用户特征列表 (可根据需要进行扩展)
        self.feature_vars = {
            "math_level": tk.DoubleVar(value=0.5),
            "science_level": tk.DoubleVar(value=0.5),
            "language_level": tk.DoubleVar(value=0.5),
            "history_level": tk.DoubleVar(value=0.5),
            "active_time": tk.IntVar(value=30)
        }
        
        feature_labels = {
            "math_level": "数学能力",
            "science_level": "科学能力",
            "language_level": "语言能力",
            "history_level": "历史能力",
            "active_time": "活跃时间(分钟/天)"
        }
        
        for key, var in self.feature_vars.items():
            frame = ttk.Frame(features_frame)
            frame.pack(fill=tk.X, padx=5, pady=3)
            
            ttk.Label(frame, text=feature_labels[key], width=15).pack(side=tk.LEFT)
            
            if key == "active_time":
                ttk.Spinbox(
                    frame, 
                    from_=0, 
                    to=240, 
                    increment=5, 
                    textvariable=var,
                    width=5
                ).pack(side=tk.LEFT)
            else:
                ttk.Scale(
                    frame, 
                    from_=0, 
                    to=1, 
                    orient=tk.HORIZONTAL, 
                    variable=var,
                    length=150
                ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 按钮区域
        button_frame = ttk.Frame(left_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=15)
        
        self.recommend_button = ttk.Button(
            button_frame, 
            text="开始推荐",
            command=self._start_recommendation,
            width=15
        )
        self.recommend_button.pack(side=tk.LEFT, padx=5)
        
        self.cancel_button = ttk.Button(
            button_frame, 
            text="取消",
            command=self._cancel_recommendation,
            width=15,
            state=tk.DISABLED
        )
        self.cancel_button.pack(side=tk.LEFT, padx=5)
        
        # 右侧：推荐结果和可视化
        result_frame = ttk.LabelFrame(main_frame, text="推荐结果")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        
        # 顶部：推荐列表
        list_frame = ttk.Frame(result_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建表格
        columns = ("id", "title", "difficulty", "score")
        self.result_table = ttk.Treeview(list_frame, columns=columns, show="headings", height=10)
        
        # 设置列标题
        self.result_table.heading("id", text="题目ID")
        self.result_table.heading("title", text="题目标题")
        self.result_table.heading("difficulty", text="难度")
        self.result_table.heading("score", text="推荐分数")
        
        # 设置列宽度
        self.result_table.column("id", width=60)
        self.result_table.column("title", width=250)
        self.result_table.column("difficulty", width=60)
        self.result_table.column("score", width=80)
        
        # 添加滚动条
        table_scroll_y = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.result_table.yview)
        self.result_table.configure(yscrollcommand=table_scroll_y.set)
        
        table_scroll_x = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL, command=self.result_table.xview)
        self.result_table.configure(xscrollcommand=table_scroll_x.set)
        
        # 布局表格和滚动条
        self.result_table.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        table_scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        table_scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 绑定表格选择事件
        self.result_table.bind("<<TreeviewSelect>>", self._on_item_selected)
        
        # 题目详情区域
        detail_frame = ttk.LabelFrame(result_frame, text="题目详情")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.detail_text = tk.Text(detail_frame, height=10, width=40, wrap=tk.WORD)
        self.detail_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 默认内容
        self.detail_text.insert("1.0", "选择题目查看详情...")
        self.detail_text.config(state=tk.DISABLED)
        
        # 推荐可视化
        viz_frame = ttk.LabelFrame(result_frame, text="推荐分布")
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 创建图表
        self.figure = Figure(figsize=(5, 3), dpi=100)
        self.plot = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 设置初始图表
        self._init_chart()
        
        # 底部按钮
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=20, pady=15)
        
        ttk.Button(
            bottom_frame, 
            text="保存推荐结果",
            command=self._export_recommendations,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        self.back_button = ttk.Button(
            button_frame,
            text="返回",
            command=lambda: self.controller.show_page("TrainPage"),
            width=15
        )
        self.back_button.pack(side=tk.RIGHT, padx=5)
    
    def _init_chart(self):
        """初始化图表"""
        self.plot.clear()
        self.plot.set_title("推荐题目分布")
        self.plot.set_xlabel("难度级别")
        self.plot.set_ylabel("数量")
        self.plot.grid(True, linestyle='--', alpha=0.7)
        self.canvas.draw()
    
    def _browse_model(self):
        """浏览选择推荐模型文件"""
        file_path = filedialog.askopenfilename(
            title="选择推荐模型",
            filetypes=[("Joblib文件", "*.joblib"), ("所有文件", "*.*")],
            defaultextension=".joblib"
        )
        if file_path:
            self.model_path.set(file_path)
    
    def _start_recommendation(self):
        """开始推荐"""
        # 参数验证
        if not self.model_path.get():
            self.show_message("请选择推荐模型文件", "warning")
            return
        
        if not os.path.exists(self.model_path.get()):
            self.show_message("推荐模型文件不存在", "error")
            return
        
        if not self.user_id.get():
            self.show_message("请输入用户ID", "warning")
            return
        
        # 更新UI状态
        self.is_recommending = True
        self.recommend_button.config(state=tk.DISABLED)
        self.cancel_button.config(state=tk.NORMAL)
        self.status_var.set("推荐中...")
        
        # 清空表格
        for item in self.result_table.get_children():
            self.result_table.delete(item)
        
        # 清空详情
        self.detail_text.config(state=tk.NORMAL)
        self.detail_text.delete("1.0", tk.END)
        self.detail_text.insert("1.0", "正在生成推荐...")
        self.detail_text.config(state=tk.DISABLED)
        
        # 重置图表
        self._init_chart()
        
        # 启动推荐线程
        self.recommending_thread = threading.Thread(
            target=self._recommendation_thread,
            daemon=True
        )
        self.recommending_thread.start()
        
        logger.info(f"开始为用户 {self.user_id.get()} 生成推荐")
    
    def _cancel_recommendation(self):
        """取消推荐"""
        if self.is_recommending:
            self.is_recommending = False
            self.status_var.set("已取消")
            
            # 恢复UI状态
            self.recommend_button.config(state=tk.NORMAL)
            self.cancel_button.config(state=tk.DISABLED)
            
            logger.info("取消推荐")
    
    def _recommendation_thread(self):
        """推荐线程函数"""
        try:
            # 获取用户ID和模型路径
            user_id = int(self.user_id.get())
            model_path = self.model_path.get()
            
            # 更新状态
            self.status_var.set("加载模型中...")
            
            # 尝试使用joblib加载模型
            try:
                import joblib
                model = joblib.load(model_path)
                logger.info(f"已加载模型: {model_path}")
                
                # 检查模型元数据文件
                metadata_path = os.path.join(os.path.dirname(model_path), "model_metadata.joblib")
                if os.path.exists(metadata_path):
                    metadata = joblib.load(metadata_path)
                    logger.info(f"已加载模型元数据: {metadata_path}")
                
                self.status_var.set("生成推荐中...")
            except Exception as e:
                logger.error(f"加载模型失败: {str(e)}")
                raise Exception(f"加载模型失败: {str(e)}")
            
            # 在真实场景下，这里应该使用加载的模型生成真实推荐
            # 但由于我们是模拟系统，继续使用模拟数据
            self._generate_mock_recommendations()
            
            # 如果过程未被取消，更新UI
            if self.is_recommending and self.winfo_exists():
                # 更新状态
                self.status_var.set(f"已为用户 {self.user_id.get()} 生成推荐")
                
                # 添加到推荐历史
                self.last_recommendations = {
                    "user_id": self.user_id.get(),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "difficulty": self.difficulty_level.get(),
                    "items": self.recommended_items,
                    "features": {k: v.get() for k, v in self.feature_vars.items()}
                }
                
                # 更新图表
                self._update_chart()
        
        except Exception as e:
            logger.error(f"推荐过程出错: {str(e)}")
            if self.winfo_exists():
                self.status_var.set(f"错误: {str(e)}")
                self.show_message(f"推荐过程出错: {str(e)}", "error")
        
        finally:
            # 恢复UI状态
            if self.winfo_exists():
                self.is_recommending = False
                self.recommend_button.config(state=tk.NORMAL)
                self.cancel_button.config(state=tk.DISABLED)
    
    def _generate_mock_recommendations(self):
        """生成模拟推荐结果"""
        # 清空现有数据
        self.recommended_items = []
        
        # 推荐题目的分类
        categories = ["数学", "物理", "化学", "生物", "历史", "地理", "语文", "英语"]
        difficulty_labels = ["非常简单", "简单", "中等", "困难", "非常困难"]
        
        # 设置用户选择的难度级别
        target_difficulty = self.difficulty_level.get()
        
        # 模拟推荐过程中的延迟
        time.sleep(1.5)
        
        # 基于用户特征生成推荐
        num_items = self.num_recommendations.get()
        
        for i in range(num_items):
            if not self.is_recommending:
                break
                
            # 模拟难度分布 - 围绕目标难度生成推荐
            diff = min(5, max(1, int(np.random.normal(target_difficulty, 0.8))))
            
            # 基于用户特征调整分类概率
            math_prob = self.feature_vars["math_level"].get()
            science_prob = self.feature_vars["science_level"].get()
            language_prob = self.feature_vars["language_level"].get()
            history_prob = self.feature_vars["history_level"].get()
            
            # 创建分类权重
            weights = [
                math_prob,                  # 数学
                science_prob * 0.5,         # 物理
                science_prob * 0.5,         # 化学
                science_prob * 0.3,         # 生物
                history_prob * 0.5,         # 历史
                history_prob * 0.5,         # 地理
                language_prob * 0.7,        # 语文
                language_prob * 0.5         # 英语
            ]
            
            # 归一化权重
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            # 选择分类
            category = np.random.choice(categories, p=weights)
            
            # 生成推荐分数 - 取决于与用户特征的匹配度
            base_score = 0.7 + 0.3 * np.random.random()
            
            # 调整分数 - 难度接近目标难度时分数更高
            difficulty_match = 1.0 - 0.15 * abs(diff - target_difficulty)
            score = base_score * difficulty_match
            
            # 创建推荐项
            item = {
                "id": 1000 + i,
                "title": f"{category}练习题 #{i+1}",
                "difficulty": diff,
                "difficulty_label": difficulty_labels[diff-1],
                "score": score,
                "category": category,
                "content": self._generate_mock_content(category, diff)
            }
            
            # 添加到推荐列表
            self.recommended_items.append(item)
            
            # 添加到表格
            if self.winfo_exists():
                self.result_table.insert("", tk.END, values=(
                    item["id"],
                    item["title"],
                    item["difficulty"],
                    f"{item['score']:.2f}"
                ))
            
            # 模拟推荐过程的延迟
            time.sleep(0.2)
    
    def _generate_mock_content(self, category, difficulty):
        """
        生成模拟题目内容
        
        Args:
            category: 题目分类
            difficulty: 难度级别
            
        Returns:
            str: 题目内容
        """
        # 根据分类和难度生成不同的模拟内容
        if category == "数学":
            if difficulty <= 2:
                return "计算: 12 × 8 + 15 ÷ 3 = ?"
            elif difficulty <= 4:
                return "求解方程: 2x² + 5x - 3 = 0"
            else:
                return "证明: 若 f'(x) = f(x)，且 f(0) = 1，求 f(x) 的表达式。"
        
        elif category == "物理":
            if difficulty <= 2:
                return "一个物体从10米高处自由落下，需要多少时间落到地面？(g=10m/s²)"
            elif difficulty <= 4:
                return "一个质量为2kg的物体在光滑斜面上滑动，斜面倾角为30°，求物体的加速度。"
            else:
                return "分析单摆的小角度运动方程，并推导出周期公式。"
        
        elif category in ["化学", "生物"]:
            difficulties = [
                "列举三种常见的化学反应类型。",
                "描述细胞呼吸的过程。",
                "解释化学平衡原理并举例说明。",
                "分析DNA复制过程中的关键酶的作用机制。",
                "论述有机化合物的立体化学特性对生物活性的影响。"
            ]
            return difficulties[min(difficulty-1, 4)]
        
        elif category in ["历史", "地理"]:
            difficulties = [
                "列举中国四大发明。",
                "描述长江流域的主要城市和经济特点。",
                "分析明清时期中国对外贸易的主要特点。",
                "比较东亚和西欧地区农业发展的异同。",
                "评价地理大发现对世界经济格局的深远影响。"
            ]
            return difficulties[min(difficulty-1, 4)]
        
        else:  # 语文/英语
            difficulties = [
                "给下列词语补充适当的量词。",
                "分析下面句子的成分。",
                "概括下面短文的中心思想。",
                "分析下面文章中运用的修辞手法及其效果。",
                "评价下面作品的艺术特色和思想内涵。"
            ]
            return difficulties[min(difficulty-1, 4)]
    
    def _on_item_selected(self, event):
        """表格项目选择事件处理"""
        # 获取选择的项目
        selection = self.result_table.selection()
        if not selection:
            return
        
        # 获取选中的索引
        index = self.result_table.index(selection[0])
        
        # 确保索引有效
        if 0 <= index < len(self.recommended_items):
            # 获取详细信息
            item = self.recommended_items[index]
            
            # 更新详情文本
            self.detail_text.config(state=tk.NORMAL)
            self.detail_text.delete("1.0", tk.END)
            
            detail_text = f"""题目ID: {item['id']}
标题: {item['title']}
分类: {item['category']}
难度: {item['difficulty_label']} ({item['difficulty']}/5)
推荐分数: {item['score']:.2f}

题目内容:
{item['content']}
            """
            
            self.detail_text.insert("1.0", detail_text)
            self.detail_text.config(state=tk.DISABLED)
    
    def _update_chart(self):
        """更新推荐分布图表"""
        if not self.recommended_items:
            return
            
        # 按难度统计推荐项
        difficulties = [item["difficulty"] for item in self.recommended_items]
        difficulty_counts = np.zeros(5)
        
        for d in difficulties:
            difficulty_counts[d-1] += 1
        
        # 按分类统计推荐项
        categories = {}
        for item in self.recommended_items:
            cat = item["category"]
            if cat not in categories:
                categories[cat] = 0
            categories[cat] += 1
        
        # 清除原图
        self.plot.clear()
        
        # 绘制难度分布
        x = np.arange(1, 6)
        self.plot.bar(x, difficulty_counts, color='skyblue', alpha=0.7)
        
        # 添加数据标签
        for i, count in enumerate(difficulty_counts):
            if count > 0:
                self.plot.text(i+1, count, str(int(count)), ha='center', va='bottom')
        
        # 设置图表属性
        self.plot.set_title("推荐题目难度分布")
        self.plot.set_xlabel("难度级别")
        self.plot.set_ylabel("题目数量")
        self.plot.set_xticks(x)
        self.plot.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 重新绘制
        self.canvas.draw()
    
    def _show_history(self):
        """显示历史推荐记录"""
        # 这里可以实现显示用户历史记录的功能
        # 在实际项目中，应该从数据库或文件中加载历史记录
        
        if not self.last_recommendations:
            self.show_message("没有历史推荐记录", "info")
            return
        
        # 创建历史窗口
        history_window = tk.Toplevel(self)
        history_window.title("推荐历史记录")
        history_window.geometry("800x600")
        history_window.minsize(700, 500)
        
        # 显示最近的推荐
        frame = ttk.Frame(history_window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(
            frame, 
            text=f"用户 {self.last_recommendations['user_id']} 的推荐历史",
            font=("Helvetica", 12, "bold")
        ).pack(pady=10)
        
        ttk.Label(
            frame, 
            text=f"时间: {self.last_recommendations['timestamp']}",
        ).pack(anchor=tk.W)
        
        ttk.Label(
            frame, 
            text=f"难度级别: {self.last_recommendations['difficulty']}",
        ).pack(anchor=tk.W)
        
        # 创建历史记录表格
        columns = ("id", "title", "difficulty", "score", "category")
        history_table = ttk.Treeview(frame, columns=columns, show="headings", height=15)
        
        # 设置列标题
        history_table.heading("id", text="题目ID")
        history_table.heading("title", text="题目标题")
        history_table.heading("difficulty", text="难度")
        history_table.heading("score", text="推荐分数")
        history_table.heading("category", text="分类")
        
        # 设置列宽度
        history_table.column("id", width=60)
        history_table.column("title", width=300)
        history_table.column("difficulty", width=60)
        history_table.column("score", width=80)
        history_table.column("category", width=150)
        
        # 添加滚动条
        table_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=history_table.yview)
        history_table.configure(yscrollcommand=table_scroll.set)
        
        # 布局
        history_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=10)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        
        # 添加数据
        for item in self.last_recommendations['items']:
            history_table.insert("", tk.END, values=(
                item["id"],
                item["title"],
                item["difficulty"],
                f"{item['score']:.2f}",
                item["category"]
            ))
        
        # 添加按钮
        button_frame = ttk.Frame(history_window)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(
            button_frame, 
            text="关闭",
            command=history_window.destroy,
            width=10
        ).pack(side=tk.RIGHT, padx=5)
    
    def _export_recommendations(self):
        """导出推荐结果"""
        if not self.recommended_items:
            self.show_message("没有推荐结果可导出", "warning")
            return
        
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="导出推荐结果",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("JSON文件", "*.json")]
        )
        
        if not file_path:
            return
        
        try:
            # 准备导出数据
            export_data = []
            
            for item in self.recommended_items:
                export_data.append({
                    "id": item["id"],
                    "title": item["title"],
                    "difficulty": item["difficulty"],
                    "difficulty_label": item["difficulty_label"],
                    "score": item["score"],
                    "category": item["category"],
                    "user_id": self.user_id.get(),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                })
            
            # 根据文件扩展名选择导出格式
            if file_path.endswith(".csv"):
                pd.DataFrame(export_data).to_csv(file_path, index=False)
            elif file_path.endswith(".xlsx"):
                pd.DataFrame(export_data).to_excel(file_path, index=False)
            elif file_path.endswith(".json"):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            self.show_message(f"推荐结果已成功导出到: {file_path}", "info")
            
        except Exception as e:
            logger.error(f"导出推荐结果出错: {str(e)}")
            self.show_message(f"导出推荐结果出错: {str(e)}", "error")
    
    def refresh(self):
        """刷新页面内容"""
        pass
    
    def on_show(self):
        """页面显示时的回调"""
        logger.debug("题目推荐页面显示")
    
    def on_hide(self):
        """页面隐藏时的回调"""
        # 取消推荐
        if self.is_recommending:
            self._cancel_recommendation()


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("题目推荐页面测试")
    root.geometry("1200x700")
    
    # 模拟控制器
    class MockController:
        def __init__(self):
            self.shared_data = {}
        
        def show_page(self, page_name):
            print(f"切换到页面: {page_name}")
    
    # 创建页面
    page = RecommendPage(root, MockController())
    page.show()
    
    # 显示窗口
    root.mainloop() 