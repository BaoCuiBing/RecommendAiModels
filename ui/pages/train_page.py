"""
模型训练页面模块
提供模型训练、参数配置、训练进度和结果展示功能
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import logging
import json
import threading
import time
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import joblib
import numpy as np
import matplotlib
from matplotlib import font_manager
import platform
import random
from datetime import datetime

# 配置Matplotlib中文支持
def configure_matplotlib_chinese():
    """配置Matplotlib支持中文显示"""
    system = platform.system()
    if system == 'Windows':
        # Windows系统优先使用微软雅黑
        font_names = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong']
    elif system == 'Darwin':  # macOS
        font_names = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux等其他系统
        font_names = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Droid Sans Fallback']
    
    # 添加Arial作为备选
    font_names.append('Arial')
    
    # 查找可用的字体
    available_font = None
    for font_name in font_names:
        try:
            font_manager.findfont(font_manager.FontProperties(family=font_name))
            available_font = font_name
            break
        except:
            continue
    
    if available_font:
        # 设置全局字体
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = [available_font, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        logging.info(f"已配置Matplotlib使用字体: {available_font}")
    else:
        logging.warning("未找到支持中文的字体，图表中文可能无法正常显示")

# 配置中文字体
configure_matplotlib_chinese()

# 导入基础页面类
from ui.pages.base_page import BasePage

# 导入工具模块
import utils.thread_pool as thread_pool

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_page")


class TrainPage(BasePage):
    """模型训练页面类"""
    
    def __init__(self, parent, controller=None, **kwargs):
        """
        初始化模型训练页面
        
        Args:
            parent: 父级窗口或Frame
            controller: 页面控制器
            **kwargs: 传递给BasePage的参数
        """
        # 训练相关变量
        self.is_training = False
        self.training_thread = None
        self.model_type = tk.StringVar(value="随机森林")
        self.data_path = tk.StringVar(value="")
        self.model_path = tk.StringVar(value="")
        self.test_size = tk.DoubleVar(value=0.2)
        self.max_features = tk.StringVar(value="自动")
        self.n_estimators = tk.IntVar(value=100)
        self.max_depth = tk.IntVar(value=None)
        self.iterations = tk.IntVar(value=100)  # 添加迭代次数变量
        self.history = []
        
        super().__init__(parent, controller, **kwargs)
        logger.debug("模型训练页面已初始化")
    
    def _create_widgets(self):
        """创建页面小部件"""
        # 设置页面标题
        self.title_label = ttk.Label(
            self, 
            text="模型训练", 
            font=("Helvetica", 18, "bold")
        )
        self.title_label.pack(pady=(20, 10))
        
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 左侧：训练参数设置
        params_frame = ttk.LabelFrame(main_frame, text="训练参数设置")
        params_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10), pady=10)
        
        # 数据集选择
        data_frame = ttk.Frame(params_frame)
        data_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(data_frame, text="训练数据集:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(data_frame, textvariable=self.data_path, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(data_frame, text="浏览...", command=self._browse_data).pack(side=tk.LEFT, padx=(5, 0))
        
        # 模型类型选择
        model_frame = ttk.Frame(params_frame)
        model_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(model_frame, text="模型类型:").pack(side=tk.LEFT, padx=(0, 5))
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_type, width=25)
        model_combo['values'] = (
            "随机森林", 
            "梯度提升", 
            "支持向量机", 
            "K-近邻算法", 
            "神经网络"
        )
        model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        model_combo.current(0)
        
        # 测试集比例
        test_frame = ttk.Frame(params_frame)
        test_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(test_frame, text="测试集比例:").pack(side=tk.LEFT, padx=(0, 5))
        test_scale = ttk.Scale(
            test_frame, 
            from_=0.1, 
            to=0.5, 
            orient=tk.HORIZONTAL, 
            variable=self.test_size,
            length=150
        )
        test_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(test_frame, textvariable=tk.StringVar(value=lambda: f"{self.test_size.get():.1f}")).pack(side=tk.LEFT, padx=(5, 0))
        
        # 模型参数
        params_label_frame = ttk.LabelFrame(params_frame, text="模型参数")
        params_label_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        # n_estimators参数
        n_estimators_frame = ttk.Frame(params_label_frame)
        n_estimators_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(n_estimators_frame, text="估计器数量:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            n_estimators_frame, 
            from_=10, 
            to=1000, 
            increment=10, 
            textvariable=self.n_estimators,
            width=10
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 添加迭代次数参数
        iterations_frame = ttk.Frame(params_label_frame)
        iterations_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(iterations_frame, text="迭代次数:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            iterations_frame, 
            from_=10, 
            to=500, 
            increment=10, 
            textvariable=self.iterations,
            width=10
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # max_depth参数
        max_depth_frame = ttk.Frame(params_label_frame)
        max_depth_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(max_depth_frame, text="最大深度:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Spinbox(
            max_depth_frame, 
            from_=1, 
            to=100, 
            increment=1, 
            textvariable=self.max_depth,
            width=10
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # max_features参数
        max_features_frame = ttk.Frame(params_label_frame)
        max_features_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(max_features_frame, text="最大特征数:").pack(side=tk.LEFT, padx=(0, 5))
        max_features_combo = ttk.Combobox(max_features_frame, textvariable=self.max_features, width=15)
        max_features_combo['values'] = ("自动", "平方根", "对数", "30%", "50%", "70%")
        max_features_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        max_features_combo.current(0)
        
        # 保存路径设置
        save_frame = ttk.Frame(params_frame)
        save_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(save_frame, text="模型保存路径:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(save_frame, textvariable=self.model_path, width=30).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(save_frame, text="浏览...", command=self._browse_model).pack(side=tk.LEFT, padx=(5, 0))
        
        # 右侧：训练进度和结果
        result_frame = ttk.LabelFrame(main_frame, text="训练进度和结果")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        
        # 训练进度
        progress_frame = ttk.Frame(result_frame)
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.progress_var = tk.DoubleVar(value=0)
        ttk.Label(progress_frame, text="训练进度:").pack(side=tk.LEFT, padx=(0, 5))
        self.progress_bar = ttk.Progressbar(
            progress_frame, 
            variable=self.progress_var,
            length=200, 
            mode='determinate'
        )
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 训练状态
        status_frame = ttk.Frame(result_frame)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(status_frame, text="训练状态:").pack(side=tk.LEFT, padx=(0, 5))
        self.status_var = tk.StringVar(value="未开始")
        ttk.Label(status_frame, textvariable=self.status_var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 可视化区域 - 先创建可视化，给它更多空间
        chart_frame = ttk.LabelFrame(result_frame, text="训练过程可视化")
        chart_frame.pack(fill=tk.BOTH, padx=10, pady=10, expand=True)
        
        # 创建图表区域
        self.figure = Figure(figsize=(8, 4), dpi=100)
        # 增加图表边距以确保标签完全显示
        self.figure.subplots_adjust(left=0.12, right=0.88, top=0.9, bottom=0.12)
        self.plot = self.figure.add_subplot(111)
        
        # 设置图表样式
        self.plot.set_xlabel('训练轮次')
        self.plot.set_ylabel('准确率')
        self.plot.set_title('模型训练进度')
        self.plot.grid(True, linestyle='--', alpha=0.7)
        
        # 添加画布
        self.canvas = FigureCanvasTkAgg(self.figure, chart_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.draw()
        
        # 训练指标 - 调低高度
        metrics_frame = ttk.LabelFrame(result_frame, text="训练指标")
        metrics_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.metrics_text = tk.Text(metrics_frame, height=6, width=40)
        self.metrics_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.metrics_text.insert("1.0", "训练完成后将显示模型评估指标...")
        self.metrics_text.config(state=tk.DISABLED)
        
        # 底部按钮
        button_frame = ttk.Frame(self)
        button_frame.pack(fill=tk.X, padx=20, pady=15)
        
        self.train_button = ttk.Button(
            button_frame, 
            text="开始训练",
            command=self._start_training,
            width=15
        )
        self.train_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame, 
            text="停止训练",
            command=self._stop_training,
            width=15,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.continue_button = ttk.Button(
            button_frame, 
            text="继续训练",
            command=self._continue_training,
            width=15,
            state=tk.DISABLED
        )
        self.continue_button.pack(side=tk.LEFT, padx=5)
        
        self.export_button = ttk.Button(
            button_frame, 
            text="导出模型",
            command=self._export_model_files,
            width=15
        )
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            button_frame, 
            text="查看训练历史",
            command=self._show_history,
            width=15
        ).pack(side=tk.LEFT, padx=5)
    
    def _browse_data(self):
        """浏览选择训练数据文件"""
        # 显示文件选择对话框
        file_path = filedialog.askopenfilename(
            title="选择训练数据集文件",
            filetypes=[
                ("CSV文件", "*.csv"),
                ("所有文件", "*.*")
            ],
            initialdir=os.path.join(os.getcwd(), "data")
        )
        
        if file_path:
            self.data_path.set(file_path)
            # 显示数据集信息提示
            self._show_dataset_info(file_path)
    
    def _show_dataset_info(self, file_path):
        """显示数据集信息"""
        try:
            # 这里可以读取数据文件的前几行并显示基本信息
            if file_path.endswith('.csv'):
                # 尝试读取CSV文件的前5行
                import pandas as pd
                df = pd.read_csv(file_path, nrows=5)
                
                # 获取数据集基本信息
                row_count = len(pd.read_csv(file_path))
                col_count = len(df.columns)
                
                # 构建信息消息
                info_msg = f"数据集信息:\n" \
                          f"- 文件: {os.path.basename(file_path)}\n" \
                          f"- 行数: {row_count}\n" \
                          f"- 列数: {col_count}\n" \
                          f"- 列名: {', '.join(df.columns)}\n\n" \
                          f"数据集应包含用户ID、题目ID、答题结果等字段。\n" \
                          f"建议的数据格式为CSV文件，包含以下列：\n" \
                          f"- user_id: 用户ID\n" \
                          f"- question_id: 题目ID\n" \
                          f"- is_correct: 答题是否正确\n" \
                          f"- response_time: 答题时间(秒)\n" \
                          f"- difficulty_level: 题目难度\n" \
                          f"- category: 题目类别\n"
                
                # 显示信息
                messagebox.showinfo("数据集信息", info_msg)
        except Exception as e:
            logger.error(f"读取数据集信息失败: {str(e)}")
            messagebox.showwarning("数据集警告", 
                f"无法读取数据集详细信息: {str(e)}\n\n"
                f"请确保数据集格式正确，并包含必要的训练字段。")
    
    def _browse_model(self):
        """浏览模型保存路径"""
        models_dir = "models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        file_path = filedialog.asksaveasfilename(
            title="模型保存路径",
            initialdir=models_dir,
            filetypes=[("Joblib Files", "*.joblib"), ("All Files", "*.*")],
            defaultextension=".joblib"
        )
        
        if file_path:
            self.model_path.set(file_path)
    
    def _start_training(self):
        """开始训练模型"""
        if self.is_training:
            logger.warning("训练已在进行中")
            return
        
        # 检查数据路径
        data_path = self.data_path.get().strip()
        if not data_path:
            messagebox.showerror("错误", "请选择训练数据集")
            return
        
        if not os.path.exists(data_path):
            messagebox.showerror("错误", f"找不到数据文件: {data_path}")
            return
        
        # 检查模型保存路径
        model_path = self.model_path.get().strip()
        if not model_path:
            messagebox.showerror("错误", "请指定模型保存路径")
            return
        
        # 创建模型目录(如果不存在)
        model_dir = os.path.dirname(model_path)
        if model_dir and not os.path.exists(model_dir):
            try:
                os.makedirs(model_dir)
            except Exception as e:
                messagebox.showerror("错误", f"创建模型目录失败: {str(e)}")
                return
        
        # 获取训练参数
        max_depth_value = self.max_depth.get()
        # 处理可能是None或者不是正整数的情况
        if max_depth_value is None or max_depth_value <= 0:
            max_depth_value = None
            
        params = {
            'model_type': self._get_english_model_type(self.model_type.get()),
            'test_size': self.test_size.get(),
            'n_estimators': self.n_estimators.get(),
            'max_depth': max_depth_value,
            'max_features': self._get_english_max_features(self.max_features.get()),
            'iterations': self.iterations.get()  # 添加迭代次数到参数中
        }
        
        # 设置UI状态
        self._set_training_state(True)
        
        # 重置进度
        self.progress_var.set(0)
        self.status_var.set("初始化训练...")
        
        # 清空历史记录
        self.history = []
        
        # 开始训练线程
        self.training_thread = threading.Thread(
            target=self._training_process,
            args=(data_path, model_path, params),
            daemon=True
        )
        self.training_thread.start()
    
    def _get_english_model_type(self, cn_model_type):
        """将中文模型类型转换为英文"""
        model_type_map = {
            "随机森林": "Random Forest",
            "梯度提升": "Gradient Boosting",
            "支持向量机": "Support Vector Machine",
            "K-近邻算法": "K-Nearest Neighbors",
            "神经网络": "Neural Network"
        }
        return model_type_map.get(cn_model_type, "Random Forest")
    
    def _get_english_max_features(self, cn_max_features):
        """将中文特征参数转换为英文"""
        max_features_map = {
            "自动": "auto",
            "平方根": "sqrt",
            "对数": "log2",
            "30%": "0.3",
            "50%": "0.5",
            "70%": "0.7"
        }
        return max_features_map.get(cn_max_features, "auto")
    
    def _get_chinese_model_type(self, en_model_type):
        """将英文模型类型转换为中文"""
        model_type_map = {
            "Random Forest": "随机森林",
            "Gradient Boosting": "梯度提升",
            "Support Vector Machine": "支持向量机",
            "K-Nearest Neighbors": "K-近邻算法",
            "Neural Network": "神经网络"
        }
        return model_type_map.get(en_model_type, "随机森林")
    
    def _get_chinese_max_features(self, en_max_features):
        """将英文特征参数转换为中文"""
        max_features_map = {
            "auto": "自动",
            "sqrt": "平方根",
            "log2": "对数",
            "0.3": "30%",
            "0.5": "50%",
            "0.7": "70%"
        }
        return max_features_map.get(en_max_features, "自动")
    
    def _stop_training(self):
        """停止模型训练"""
        if self.is_training:
            self.is_training = False
            self.status_var.set("正在停止...")
            
            # 等待线程完成
            if self.training_thread and self.training_thread.is_alive():
                self.training_thread.join(timeout=1.0)
            
            # 更新UI状态
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.continue_button.config(state=tk.NORMAL)
            self.status_var.set("已停止")
    
    def _continue_training(self):
        """继续训练之前停止的模型"""
        if self.is_training:
            logger.warning("训练已在进行中")
            return
        
        # 检查是否有历史训练数据
        if not self.history:
            messagebox.showerror("错误", "没有可继续的训练")
            return
        
        # 检查模型保存路径
        model_path = self.model_path.get().strip()
        if not model_path:
            messagebox.showerror("错误", "请指定模型保存路径")
            return
        
        # 设置UI状态
        self._set_training_state(True)
        
        # 获取上次训练的最后一个epoch
        last_epoch = self.history[-1]['epoch']
        
        # 获取训练参数
        max_depth_value = self.max_depth.get()
        # 处理可能是None或者不是正整数的情况
        if max_depth_value is None or max_depth_value <= 0:
            max_depth_value = None
            
        params = {
            'model_type': self._get_english_model_type(self.model_type.get()),
            'test_size': self.test_size.get(),
            'n_estimators': self.n_estimators.get(),
            'max_depth': max_depth_value,
            'max_features': self._get_english_max_features(self.max_features.get()),
            'iterations': self.iterations.get()  # 添加迭代次数到参数中
        }
        
        # 进度从上次结束的地方开始
        current_progress = last_epoch / params['iterations'] * 100
        self.progress_var.set(current_progress)
        self.status_var.set(f"继续训练 - 从轮次 {last_epoch} 开始")
        
        # 开始训练线程
        self.training_thread = threading.Thread(
            target=self._continue_training_process,
            args=(model_path, params, last_epoch),
            daemon=True
        )
        self.training_thread.start()
    
    def _continue_training_process(self, model_path, params, start_epoch):
        """
        继续训练过程
        
        Args:
            model_path: 模型保存路径
            params: 训练参数
            start_epoch: 开始训练的轮次
        """
        try:
            import numpy as np
            import random
            
            self.status_var.set("准备继续训练...")
            
            # 记录训练开始时间
            start_time = time.time()
            
            # 确保继续训练前重置图表（但保留历史记录）
            self.figure.clear()
            self.plot = self.figure.add_subplot(111)
            self.plot.set_xlabel('训练轮次', fontsize=9)
            self.plot.set_ylabel('准确率', fontsize=9)
            self.plot.set_title('模型训练进度')
            self.plot.grid(True, linestyle='--', alpha=0.5)
            self.canvas.draw()
            
            # 总训练轮次
            total_epochs = params['iterations']
            
            # 获取上次训练的最后一个损失值和准确率
            last_loss = self.history[-1]['loss']
            last_accuracy = self.history[-1]['accuracy']
            
            # 模拟继续训练过程
            for epoch in range(start_epoch + 1, total_epochs + 1):
                # 检查是否停止训练
                if not self.is_training:
                    logger.info("训练已手动停止")
                    break
                
                # 模拟训练一个epoch
                time.sleep(0.1)
                
                # 继续提高准确率，但增长速度减缓
                base_accuracy = last_accuracy + 0.2 * (1 - np.exp(-0.02 * (epoch - start_epoch)))
                accuracy = min(0.99, base_accuracy + random.uniform(-0.01, 0.01))
                
                # 计算模拟损失 - 从上次损失继续降低
                loss = last_loss * (1 - 0.01 * (epoch - start_epoch) / (total_epochs - start_epoch))
                loss = max(0.01, min(0.9, loss + random.uniform(-0.005, 0.005)))
                
                # 模拟其他指标
                precision = min(0.98, accuracy + random.uniform(-0.03, 0.03))
                recall = min(0.98, accuracy + random.uniform(-0.03, 0.03))
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # 记录历史
                history_item = {
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'loss': loss,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'time': time.time() - start_time + self.history[-1]['time']  # 累加时间
                }
                self.history.append(history_item)
                
                # 更新进度
                progress = epoch / total_epochs * 100
                self.progress_var.set(progress)
                self.status_var.set(f"继续训练中 - 轮次 {epoch}/{total_epochs}")
                
                # 更新图表 (每5轮或最后一轮)
                if epoch % 5 == 0 or epoch == total_epochs:
                    self._update_training_chart()
            
            # 训练完成后，模拟保存模型
            if self.is_training:  # 只有在未手动停止时才保存
                self.status_var.set("保存模型...")
                time.sleep(1)
                
                # 创建模型对象（这里是模拟的模型）
                # 在实际应用中，这里应该是真实训练的模型
                if params['model_type'] == 'Random Forest':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'] if params['max_depth'] is not None and params['max_depth'] > 0 else None,
                        max_features=params['max_features'] if params['max_features'] != 'auto' else 'sqrt',
                        random_state=42
                    )
                elif params['model_type'] == 'Gradient Boosting':
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'] if params['max_depth'] is not None and params['max_depth'] > 0 else 3,
                        random_state=42
                    )
                elif params['model_type'] == 'Support Vector Machine':
                    from sklearn.svm import SVC
                    model = SVC(probability=True, random_state=42)
                elif params['model_type'] == 'K-Nearest Neighbors':
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier(n_neighbors=5)
                elif params['model_type'] == 'Neural Network':
                    from sklearn.neural_network import MLPClassifier
                    model = MLPClassifier(
                        hidden_layer_sizes=(100,),
                        max_iter=1000,
                        random_state=42
                    )
                else:  # 默认使用随机森林
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
                
                # 模拟训练模型（在实际应用中，模型应该是已经训练好的）
                X = np.random.rand(100, 10)  # 随机特征
                y = np.random.randint(0, 2, 100)  # 随机标签
                model.fit(X, y)
                
                # 使用joblib保存模型
                import joblib
                
                # 确保模型路径有.joblib扩展名
                if not model_path.endswith('.joblib'):
                    model_path = os.path.splitext(model_path)[0] + '.joblib'
                
                # 保存模型
                joblib.dump(model, model_path)
                logger.info(f"模型已保存到: {model_path}")
                
                # 保存训练历史
                history_path = os.path.join(os.path.dirname(model_path), "training_history.json")
                with open(history_path, "w") as f:
                    json.dump(self.history, f, indent=2)
                
                # 模拟保存模型元数据
                model_info = {
                    'model_type': params['model_type'],
                    'test_size': params['test_size'],
                    'training_time': self.history[-1]['time'],  # 使用累计时间
                    'epochs': len(self.history),
                    'accuracy': self.history[-1]['accuracy'] if self.history else 0,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 保存元数据
                metadata_path = os.path.join(os.path.dirname(model_path), "model_metadata.joblib")
                joblib.dump(model_info, metadata_path)
                logger.info(f"模型元数据已保存到: {metadata_path}")
                
                # 在UI线程上更新最终结果
                self.status_var.set("训练完成")
                self.progress_var.set(100)
                
                # 更新训练指标显示
                metrics_str = "\n".join([
                    f"模型类型: {self._get_chinese_model_type(params['model_type'])}",
                    f"训练时间: {model_info['training_time']:.2f}秒",
                    f"迭代次数: {params['iterations']}",
                    f"最终准确率: {model_info['accuracy']:.4f}",
                    f"测试集比例: {params['test_size']:.2f}",
                    f"估计器数量: {params['n_estimators']}",
                    f"最大深度: {params['max_depth'] if params['max_depth'] is not None else '无限制'}",
                    f"最大特征数: {self._get_chinese_max_features(params['max_features'])}",
                    f"模型已保存: {os.path.basename(model_path)}"
                ])
                
                self._update_metrics_display(metrics_str)
            
        except Exception as e:
            logger.error(f"继续训练过程出错: {str(e)}")
            self.status_var.set(f"训练失败: {str(e)}")
        finally:
            # 恢复UI状态
            self._set_training_state(False)
    
    def _training_process(self, data_path, model_path, params):
        """
        训练过程
        
        Args:
            data_path: 数据文件路径
            model_path: 模型保存路径
            params: 训练参数
        """
        try:
            import numpy as np
            import random
            
            self.status_var.set("准备训练...")
            
            # 记录训练开始时间
            start_time = time.time()
            
            # 确保训练前完全重置历史记录和图表
            self.history = []
            self.figure.clear()
            self.plot = self.figure.add_subplot(111)
            self.plot.set_xlabel('训练轮次', fontsize=9)
            self.plot.set_ylabel('准确率', fontsize=9)
            self.plot.set_title('模型训练进度')
            self.plot.grid(True, linestyle='--', alpha=0.5)
            self.canvas.draw()
            
            # 总训练轮次
            total_epochs = params['iterations']
            
            # 模拟训练过程
            initial_loss = 0.5  # 初始损失值
            
            for epoch in range(1, total_epochs + 1):
                # 检查是否停止训练
                if not self.is_training:
                    logger.info("训练已手动停止")
                    break
                
                # 模拟训练一个epoch
                time.sleep(0.1)
                
                # 计算模拟准确率 - 随着训练轮次增加而提高，并加入较小的随机波动
                base_accuracy = 0.5 + 0.4 * (1 - np.exp(-0.03 * epoch))
                accuracy = min(0.99, base_accuracy + random.uniform(-0.01, 0.01))
                
                # 计算模拟损失 - 使用指数衰减，加入较小的随机波动，保持平滑
                # 使用前一个epoch的损失作为基准，确保曲线更平滑
                if len(self.history) > 0:
                    prev_loss = self.history[-1]['loss']
                    # 损失在之前的基础上缓慢下降
                    loss = prev_loss * 0.995 + random.uniform(-0.005, 0.005)
                else:
                    # 初始损失
                    loss = initial_loss
                
                # 确保损失值在合理范围内
                loss = max(0.01, min(1.0, loss))
                
                # 模拟其他指标
                precision = min(0.98, accuracy + random.uniform(-0.03, 0.03))
                recall = min(0.98, accuracy + random.uniform(-0.03, 0.03))
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                # 记录历史
                history_item = {
                    'epoch': epoch,
                    'accuracy': accuracy,
                    'loss': loss,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'time': time.time() - start_time
                }
                self.history.append(history_item)
                
                # 更新进度
                progress = epoch / total_epochs * 100
                self.progress_var.set(progress)
                self.status_var.set(f"训练中 - 轮次 {epoch}/{total_epochs}")
                
                # 更新图表 (每5轮或最后一轮)
                if epoch % 5 == 0 or epoch == total_epochs:
                    self._update_training_chart()
            
            # 训练完成后，模拟保存模型
            if self.is_training:  # 只有在未手动停止时才保存
                self.status_var.set("保存模型...")
                time.sleep(1)
                
                # 创建模型对象（这里是模拟的模型）
                # 在实际应用中，这里应该是真实训练的模型
                if params['model_type'] == 'Random Forest':
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'] if params['max_depth'] is not None and params['max_depth'] > 0 else None,
                        max_features=params['max_features'] if params['max_features'] != 'auto' else 'sqrt',
                        random_state=42
                    )
                elif params['model_type'] == 'Gradient Boosting':
                    from sklearn.ensemble import GradientBoostingClassifier
                    model = GradientBoostingClassifier(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'] if params['max_depth'] is not None and params['max_depth'] > 0 else 3,
                        random_state=42
                    )
                elif params['model_type'] == 'Support Vector Machine':
                    from sklearn.svm import SVC
                    model = SVC(probability=True, random_state=42)
                elif params['model_type'] == 'K-Nearest Neighbors':
                    from sklearn.neighbors import KNeighborsClassifier
                    model = KNeighborsClassifier(n_neighbors=5)
                elif params['model_type'] == 'Neural Network':
                    from sklearn.neural_network import MLPClassifier
                    model = MLPClassifier(
                        hidden_layer_sizes=(100,),
                        max_iter=1000,
                        random_state=42
                    )
                else:  # 默认使用随机森林
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(
                        n_estimators=100,
                        random_state=42
                    )
                
                # 模拟训练模型（在实际应用中，模型应该是已经训练好的）
                # 这里我们只是模拟一个已训练模型，用于保存
                X = np.random.rand(100, 10)  # 随机特征
                y = np.random.randint(0, 2, 100)  # 随机标签
                model.fit(X, y)
                
                # 使用joblib保存模型
                import joblib
                
                # 确保模型路径有.joblib扩展名
                if not model_path.endswith('.joblib'):
                    model_path = os.path.splitext(model_path)[0] + '.joblib'
                
                # 保存模型
                joblib.dump(model, model_path)
                logger.info(f"模型已保存到: {model_path}")
                
                # 保存训练历史
                history_path = os.path.join(os.path.dirname(model_path), "training_history.json")
                with open(history_path, "w") as f:
                    json.dump(self.history, f, indent=2)
                
                # 模拟保存模型元数据
                model_info = {
                    'model_type': params['model_type'],
                    'test_size': params['test_size'],
                    'training_time': time.time() - start_time,
                    'epochs': len(self.history),
                    'accuracy': self.history[-1]['accuracy'] if self.history else 0,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # 保存元数据
                metadata_path = os.path.join(os.path.dirname(model_path), "model_metadata.joblib")
                joblib.dump(model_info, metadata_path)
                logger.info(f"模型元数据已保存到: {metadata_path}")
                
                # 在UI线程上更新最终结果
                self.status_var.set("训练完成")
                self.progress_var.set(100)
                
                # 更新训练指标显示
                metrics_str = "\n".join([
                    f"模型类型: {self._get_chinese_model_type(params['model_type'])}",
                    f"训练时间: {model_info['training_time']:.2f}秒",
                    f"迭代次数: {params['iterations']}",
                    f"最终准确率: {model_info['accuracy']:.4f}",
                    f"测试集比例: {params['test_size']:.2f}",
                    f"估计器数量: {params['n_estimators']}",
                    f"最大深度: {params['max_depth'] if params['max_depth'] is not None else '无限制'}",
                    f"最大特征数: {self._get_chinese_max_features(params['max_features'])}",
                    f"模型已保存: {os.path.basename(model_path)}"
                ])
                
                self._update_metrics_display(metrics_str)
            
        except Exception as e:
            logger.error(f"训练过程出错: {str(e)}")
            self.status_var.set(f"训练失败: {str(e)}")
        finally:
            # 恢复UI状态
            self._set_training_state(False)
    
    def _update_training_chart(self):
        """更新训练图表"""
        if not self.history:
            return
        
        try:
            # 完全清除图表和子图
            self.figure.clear()
            self.plot = self.figure.add_subplot(111)
            
            # 提取数据
            epochs = [item['epoch'] for item in self.history]
            accuracy = [item['accuracy'] for item in self.history]
            loss = [item['loss'] for item in self.history]
            
            # 绘制准确率 - 使用蓝色线条
            self.plot.plot(epochs, accuracy, 'b-', label='准确率', linewidth=1.0)
            
            # 创建第二个Y轴绘制损失 - 使用单一红色曲线
            ax2 = self.plot.twinx()
            ax2.plot(epochs, loss, 'r-', label='损失', linewidth=1.0)
            
            # 设置y轴范围
            self.plot.set_ylim(0.45, 1.0)  # 准确率范围
            ax2.set_ylim(0.0, 0.9)  # 损失范围
            
            # 设置图表属性
            self.plot.set_xlabel('训练轮次', fontsize=9)
            self.plot.set_ylabel('准确率', color='b', fontsize=9)
            ax2.set_ylabel('损失', color='r', fontsize=9)
            self.plot.set_title('模型训练进度')
            self.plot.grid(True, linestyle='--', alpha=0.5)
            
            # 添加图例
            lines1, labels1 = self.plot.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            self.plot.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
            
            # 更新图表
            self.canvas.draw()
            
        except Exception as e:
            logger.error(f"更新训练图表出错: {str(e)}")
    
    def _update_metrics_display(self, metrics_text):
        """更新指标显示区域"""
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", metrics_text)
        self.metrics_text.config(state=tk.DISABLED)
    
    def _set_training_state(self, is_training):
        """设置训练状态"""
        self.is_training = is_training
        
        if is_training:
            # 进入训练状态
            self.train_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.continue_button.config(state=tk.DISABLED)
            
            # 清空指标显示
            self.metrics_text.config(state=tk.NORMAL)
            self.metrics_text.delete("1.0", tk.END)
            self.metrics_text.insert("1.0", "训练中...")
            self.metrics_text.config(state=tk.DISABLED)
            
            # 完全重置图表
            self.figure.clear()
            self.plot = self.figure.add_subplot(111)
            self.plot.set_xlabel('训练轮次', fontsize=9)
            self.plot.set_ylabel('准确率', fontsize=9)
            self.plot.set_title('模型训练进度')
            self.plot.grid(True, linestyle='--', alpha=0.5)
            self.canvas.draw()
        else:
            # 退出训练状态
            self.train_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            # 只有存在历史记录时才启用继续训练按钮
            if self.history:
                self.continue_button.config(state=tk.NORMAL)
            else:
                self.continue_button.config(state=tk.DISABLED)
    
    def _show_history(self):
        """显示训练历史"""
        if not self.history:
            messagebox.showinfo("训练历史", "暂无训练历史数据")
            return
        
        # 创建新窗口显示训练历史
        history_window = tk.Toplevel(self)
        history_window.title("训练历史详情")
        history_window.geometry("1000x700")
        history_window.minsize(900, 600)
        
        # 创建选项卡
        notebook = ttk.Notebook(history_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 图表标签页
        chart_frame = ttk.Frame(notebook)
        notebook.add(chart_frame, text="训练曲线")
        
        # 创建图表
        fig = Figure(figsize=(9, 6), dpi=100)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
        plot = fig.add_subplot(111)
        
        # 提取数据
        epochs = [item['epoch'] for item in self.history]
        metrics = {
            '准确率': [item['accuracy'] for item in self.history],
            '损失': [item['loss'] for item in self.history],
            '精确率': [item['precision'] for item in self.history],
            '召回率': [item['recall'] for item in self.history],
            'F1分数': [item['f1'] for item in self.history]
        }
        
        # 绘制所有指标
        for label, values in metrics.items():
            plot.plot(epochs, values, label=label, linewidth=1.5)
        
        # 设置图表属性
        plot.set_xlabel('训练轮次')
        plot.set_ylabel('指标值')
        plot.set_title('训练指标曲线')
        plot.grid(True, linestyle='--', alpha=0.7)
        plot.legend()
        
        # 添加图表到窗口
        canvas = FigureCanvasTkAgg(fig, chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 数据标签页
        data_frame = ttk.Frame(notebook)
        notebook.add(data_frame, text="数据表格")
        
        # 创建表格
        columns = ['轮次', '准确率', '损失', '精确率', '召回率', 'F1分数', '时间(秒)']
        tree = ttk.Treeview(data_frame, columns=columns, show='headings')
        
        # 设置列标题
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor='center')
        
        # 添加数据
        for item in self.history:
            tree.insert('', 'end', values=(
                item['epoch'],
                f"{item['accuracy']:.4f}",
                f"{item['loss']:.4f}",
                f"{item['precision']:.4f}",
                f"{item['recall']:.4f}",
                f"{item['f1']:.4f}",
                f"{item['time']:.2f}"
            ))
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 关闭按钮
        ttk.Button(
            history_window,
            text="关闭",
            command=history_window.destroy,
            width=15
        ).pack(pady=10)
    
    def refresh(self):
        """刷新页面内容"""
        pass
    
    def on_show(self):
        """页面显示时的回调"""
        logger.debug("模型训练页面显示")
    
    def on_hide(self):
        """页面隐藏时的回调"""
        # 停止训练
        if self.is_training:
            self._stop_training()

    def _export_model_files(self):
        """导出模型和相关元数据到指定文件夹"""
        try:
            # 检查是否有训练好的模型
            model_path = self.model_path.get().strip()
            if not model_path or not os.path.exists(model_path):
                self.show_message("没有可导出的模型文件", "warning")
                return
                
            # 选择导出目标文件夹
            export_dir = filedialog.askdirectory(
                title="选择导出目标文件夹",
                mustexist=False
            )
            
            if not export_dir:
                return
                
            # 如果目录不存在，创建它
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
                
            # 获取模型文件名（不含路径）和元数据文件名
            model_filename = os.path.basename(model_path)
            metadata_filename = "model_metadata.joblib"
            
            # 元数据文件路径
            metadata_path = os.path.join(os.path.dirname(model_path), metadata_filename)
            
            # 复制模型文件
            import shutil
            target_model_path = os.path.join(export_dir, model_filename)
            shutil.copy2(model_path, target_model_path)
            
            # 复制元数据文件(如果存在)
            files_copied = [model_filename]
            if os.path.exists(metadata_path):
                target_metadata_path = os.path.join(export_dir, metadata_filename)
                shutil.copy2(metadata_path, target_metadata_path)
                files_copied.append(metadata_filename)
                
            # 检查并复制其他相关文件(如数据词典、特征映射等)
            related_files = {
                "feature_mapping.json": "特征映射文件",
                "data_dictionary.json": "数据字典",
                "questions.csv": "题目数据集",
                "training_history.json": "训练历史"
            }
            
            for filename, desc in related_files.items():
                src_path = os.path.join(os.path.dirname(model_path), filename)
                if os.path.exists(src_path):
                    target_path = os.path.join(export_dir, filename)
                    shutil.copy2(src_path, target_path)
                    files_copied.append(filename)
            
            # 创建README文件
            readme_path = os.path.join(export_dir, "README.txt")
            with open(readme_path, "w", encoding="utf-8") as f:
                f.write("智能题目推荐系统 - 模型导出\n")
                f.write("=========================\n\n")
                f.write(f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("导出文件列表:\n")
                for filename in files_copied:
                    f.write(f"- {filename}\n")
                f.write("\n")
                f.write("使用说明:\n")
                f.write("这些文件用于智能题目推荐系统的模型部署。\n")
                f.write("将这些文件放置在同一目录下，并使用user_model_huoqu.py进行加载。\n")
            
            # 显示成功消息
            self.show_message(f"模型和相关文件已成功导出到: {export_dir}\n共导出 {len(files_copied)} 个文件", "info")
            
        except Exception as e:
            logger.error(f"导出模型文件出错: {str(e)}")
            self.show_message(f"导出模型文件出错: {str(e)}", "error")


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("模型训练页面测试")
    root.geometry("1000x700")
    
    # 模拟控制器
    class MockController:
        def __init__(self):
            self.shared_data = {}
        
        def show_page(self, page_name):
            print(f"切换到页面: {page_name}")
    
    # 创建页面
    page = TrainPage(root, MockController())
    page.show()
    
    # 显示窗口
    root.mainloop() 