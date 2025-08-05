#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据可视化模块
用于在Tkinter界面中展示Matplotlib和Seaborn图表
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 避免Tkinter和Matplotlib的冲突
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
import logging
import threading
from datetime import datetime

# 加载系统字体
from matplotlib import font_manager
import platform

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("visualizer")

# 设置Seaborn样式
sns.set_style("whitegrid")

# 根据操作系统配置中文字体
system = platform.system()
if system == 'Windows':
    # 尝试加载Windows字体
    try:
        font_paths = font_manager.findSystemFonts()
        chinese_fonts = [f for f in font_paths if os.path.basename(f).startswith(('msyh', 'simhei', 'simsun'))]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = [font_manager.FontProperties(fname=chinese_fonts[0]).get_name()]
        else:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 默认黑体
    except Exception as e:
        logger.warning(f"加载Windows中文字体失败: {str(e)}")
        plt.rcParams['font.sans-serif'] = ['SimHei']
elif system == 'Linux':
    # 尝试加载Linux字体
    try:
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'DejaVu Sans']
    except Exception as e:
        logger.warning(f"加载Linux中文字体失败: {str(e)}")
elif system == 'Darwin':  # macOS
    # 尝试加载macOS字体
    try:
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    except Exception as e:
        logger.warning(f"加载macOS中文字体失败: {str(e)}")

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 设置精致小巧的图表样式
plt.rcParams['font.size'] = 10  # 略微增大字体
plt.rcParams['axes.linewidth'] = 0.8  # 轴线宽度
plt.rcParams['xtick.major.width'] = 0.8  # X轴刻度线宽度
plt.rcParams['ytick.major.width'] = 0.8  # Y轴刻度线宽度
plt.rcParams['lines.linewidth'] = 1.5  # 增加线条宽度使图表更清晰 
plt.rcParams['grid.linewidth'] = 0.6  # 网格线宽度
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.subplot.left'] = 0.12   # 左边距
plt.rcParams['figure.subplot.right'] = 0.95  # 右边距
plt.rcParams['figure.subplot.bottom'] = 0.12 # 底部边距
plt.rcParams['figure.subplot.top'] = 0.92    # 顶部边距


class FigureManager:
    """图表管理器，处理图表的创建和缓存"""
    
    def __init__(self):
        """初始化图表管理器"""
        self.figures = {}  # 缓存的图表 {figure_id: Figure}
        self.lock = threading.RLock()  # 线程锁
    
    def create_figure(self, figure_id: str, figsize: Tuple[float, float] = (6, 4), dpi: int = 100) -> Figure:
        """
        创建或获取图表
        
        Args:
            figure_id: 图表ID
            figsize: 图表大小 (宽, 高)，单位为英寸
            dpi: 图表DPI
            
        Returns:
            Figure: Matplotlib图表对象
        """
        with self.lock:
            if figure_id in self.figures:
                # 清除现有图表内容
                self.figures[figure_id].clear()
                return self.figures[figure_id]
            
            # 创建新图表
            fig = Figure(figsize=figsize, dpi=dpi)
            self.figures[figure_id] = fig
            return fig
    
    def remove_figure(self, figure_id: str) -> bool:
        """
        移除缓存的图表
        
        Args:
            figure_id: 图表ID
            
        Returns:
            bool: 是否成功移除
        """
        with self.lock:
            if figure_id in self.figures:
                del self.figures[figure_id]
                return True
            return False
    
    def clear_all(self) -> None:
        """清空所有缓存的图表"""
        with self.lock:
            self.figures.clear()


# 全局图表管理器实例
_figure_manager = FigureManager()


class ChartPanel(ttk.Frame):
    """图表面板组件，用于在Tkinter界面中显示Matplotlib/Seaborn图表"""
    
    def __init__(self, parent, figure_id: str = None, figsize: Tuple[float, float] = (6, 4), 
                dpi: int = 100, toolbar: bool = True, **kwargs):
        """
        初始化图表面板
        
        Args:
            parent: 父级Tkinter窗口或Frame
            figure_id: 图表ID，用于缓存和引用
            figsize: 图表大小 (宽, 高)，单位为英寸
            dpi: 图表DPI
            toolbar: 是否显示导航工具栏
            **kwargs: 传递给ttk.Frame的参数
        """
        super().__init__(parent, **kwargs)
        
        # 配置
        self.figure_id = figure_id or f"chart_{id(self)}"
        self.figsize = figsize
        self.dpi = dpi
        self.show_toolbar = toolbar
        
        # 创建图表
        self.figure = _figure_manager.create_figure(self.figure_id, figsize, dpi)
        
        # 创建Tkinter Canvas，显示图表
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # 添加工具栏（如果需要）
        if self.show_toolbar:
            self.toolbar = NavigationToolbar2Tk(self.canvas, self)
            self.toolbar.update()
        
        # 布局
        self.pack(fill=tk.BOTH, expand=True)
    
    def get_figure(self) -> Figure:
        """
        获取图表对象
        
        Returns:
            Figure: Matplotlib图表对象
        """
        return self.figure
    
    def clear(self) -> None:
        """清除图表内容"""
        self.figure.clear()
        self.canvas.draw()
    
    def draw(self) -> None:
        """重绘图表"""
        self.canvas.draw()
    
    def save_figure(self, filepath: str, dpi: int = None, format: str = None) -> bool:
        """
        保存图表为图片
        
        Args:
            filepath: 保存路径
            dpi: 保存的DPI，None表示使用图表当前DPI
            format: 保存格式，None表示从文件扩展名推断
            
        Returns:
            bool: 是否保存成功
        """
        try:
            self.figure.savefig(filepath, dpi=dpi, format=format, bbox_inches='tight')
            logger.info(f"图表已保存为: {filepath}")
            return True
        except Exception as e:
            logger.error(f"保存图表失败: {str(e)}")
            return False


# 常用绘图函数

def draw_error_distribution(chart: ChartPanel, errors: List[float], title: str = "预测误差分布") -> None:
    """
    绘制预测误差分布图
    
    Args:
        chart: 图表面板
        errors: 误差数据列表
        title: 图表标题
    """
    fig = chart.get_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    
    # 绘制直方图和密度曲线
    sns.histplot(errors, kde=True, ax=ax, color='cornflowerblue')
    
    # 添加垂直线表示均值
    mean_error = np.mean(errors)
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=1, label=f'均值: {mean_error:.4f}')
    
    # 添加标题和标签
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("误差", fontsize=9)
    ax.set_ylabel("频次", fontsize=9)
    ax.legend(fontsize=8)
    
    # 格式化轴
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 刷新图表
    chart.draw()

def draw_feature_importance(chart: ChartPanel, features: Dict[str, float], title: str = "特征重要性", top_n: int = 10) -> None:
    """
    绘制特征重要性柱状图
    
    Args:
        chart: 图表面板
        features: 特征重要性字典 {特征名: 重要性值}
        title: 图表标题
        top_n: 显示前N个最重要的特征
    """
    fig = chart.get_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    
    # 排序并获取前N个特征
    sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:top_n]
    feature_names = [item[0] for item in sorted_features]
    importance_values = [item[1] for item in sorted_features]
    
    # 将过长的特征名称进行截断
    feature_names = [name[:20] + '...' if len(name) > 20 else name for name in feature_names]
    
    # 创建水平条形图
    bars = ax.barh(range(len(feature_names)), importance_values, color='skyblue', height=0.6)
    
    # 添加标题和标签
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("重要性", fontsize=9)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        value = bar.get_width()
        ax.text(value + 0.002, bar.get_y() + bar.get_height()/2, f"{value:.4f}", 
                va='center', fontsize=7)
    
    # 调整布局
    ax.invert_yaxis()  # 反转Y轴，使最重要的特征显示在顶部
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 刷新图表
    fig.tight_layout()
    chart.draw()

def draw_confusion_matrix(chart: ChartPanel, conf_matrix: List[List[int]], 
                         labels: List[str] = None, title: str = "混淆矩阵") -> None:
    """
    绘制混淆矩阵热力图
    
    Args:
        chart: 图表面板
        conf_matrix: 混淆矩阵数据，二维列表
        labels: 标签列表，默认为None
        title: 图表标题
    """
    fig = chart.get_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    
    # 如果没有提供标签，使用数字标签
    if labels is None:
        n_classes = len(conf_matrix)
        labels = [str(i) for i in range(n_classes)]
    
    # 创建混淆矩阵热力图
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax,
               xticklabels=labels, yticklabels=labels)
    
    # 添加标题和标签
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("预测", fontsize=9)
    ax.set_ylabel("实际", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 刷新图表
    fig.tight_layout()
    chart.draw()

def draw_training_time(chart: ChartPanel, phases: List[str], times: List[float], 
                      title: str = "训练耗时分布") -> None:
    """
    绘制训练耗时分布图
    
    Args:
        chart: 图表面板
        phases: 训练阶段列表
        times: 各阶段耗时列表
        title: 图表标题
    """
    fig = chart.get_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    
    # 将过长的阶段名称进行截断
    phases = [name[:15] + '...' if len(name) > 15 else name for name in phases]
    
    # 创建条形图
    bars = ax.bar(phases, times, color='lightgreen')
    
    # 添加标题和标签
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("训练阶段", fontsize=9)
    ax.set_ylabel("耗时 (秒)", fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.xticks(rotation=45)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1, f"{height:.1f}s", 
                ha='center', va='bottom', fontsize=7)
    
    # 去除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 刷新图表
    fig.tight_layout()
    chart.draw()

def draw_algorithm_comparison(chart: ChartPanel, algorithms: Dict[str, Dict[str, float]], 
                             metric: str = "accuracy", title: str = None) -> None:
    """
    绘制算法比较箱线图
    
    Args:
        chart: 图表面板
        algorithms: 算法性能数据 {算法名: {指标1: 值1, 指标2: 值2, ...}}
        metric: 要比较的指标名称
        title: 图表标题，默认为None
    """
    if title is None:
        title = f"算法{metric}比较"
    
    fig = chart.get_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    
    # 准备数据
    data = []
    labels = []
    for alg_name, metrics in algorithms.items():
        if metric in metrics:
            data.append(metrics[metric])
            labels.append(alg_name)
    
    # 绘制条形图
    bars = ax.bar(labels, data, color='salmon')
    
    # 添加标题和标签
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("算法", fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, f"{height:.4f}", 
                ha='center', va='bottom', fontsize=7)
    
    # 去除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 刷新图表
    fig.tight_layout()
    chart.draw()

def draw_training_history(chart: ChartPanel, history: List[Dict[str, Any]], 
                         metric: str = "accuracy", title: str = None) -> None:
    """
    绘制训练历史趋势图
    
    Args:
        chart: 图表面板
        history: 训练历史数据列表
        metric: 要显示的指标
        title: 图表标题，默认为None
    """
    if title is None:
        title = f"{metric}历史趋势"
    
    fig = chart.get_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    
    # 准备数据
    timestamps = [entry.get('timestamp', '') for entry in history]
    values = [entry.get(metric, 0) for entry in history]
    
    # 格式化时间戳
    x_labels = []
    for ts in timestamps:
        try:
            dt = datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
            x_labels.append(dt.strftime('%m-%d %H:%M'))
        except (ValueError, TypeError):
            x_labels.append(ts)
    
    # 绘制折线图
    ax.plot(x_labels, values, marker='o', linestyle='-', color='teal', markersize=4)
    
    # 添加标题和标签
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("时间", fontsize=9)
    ax.set_ylabel(metric, fontsize=9)
    ax.tick_params(axis='both', which='major', labelsize=8)
    plt.xticks(rotation=45)
    
    # 去除上边框和右边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 格式化y轴刻度
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.4f'))
    
    # 刷新图表
    fig.tight_layout()
    chart.draw()

def draw_training_progress(chart: ChartPanel, current: int, total: int, 
                          stage: str = "", title: str = "训练进度") -> None:
    """
    绘制训练进度条
    
    Args:
        chart: 图表面板
        current: 当前进度
        total: 总进度
        stage: 当前阶段
        title: 图表标题
    """
    fig = chart.get_figure()
    fig.clear()
    ax = fig.add_subplot(111)
    
    # 计算进度百分比
    progress = current / total if total > 0 else 0
    percentage = progress * 100
    
    # 创建进度条
    ax.barh(0, progress, color='dodgerblue', height=0.5)
    ax.barh(0, 1, color='lightgray', height=0.5, alpha=0.3)
    
    # 添加文本标签
    ax.text(0.5, 0, f"{percentage:.1f}%", ha='center', va='center', fontsize=10, color='black')
    
    # 调整布局
    ax.set_title(f"{title}: {stage}", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 刷新图表
    chart.draw()


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("图表测试")
    root.geometry("800x600")
    
    # 创建图表面板
    chart_frame = ttk.Frame(root)
    chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    chart = ChartPanel(chart_frame, figsize=(8, 5))
    
    # 测试特征重要性图
    features = {
        "user_id": 0.15,
        "question_difficulty": 0.25,
        "user_accuracy": 0.18,
        "category_score": 0.12,
        "response_time": 0.08,
        "attempts_count": 0.05,
        "hint_usage": 0.04,
        "question_category": 0.07,
        "user_level": 0.03,
        "confidence_level": 0.03
    }
    draw_feature_importance(chart, features)
    
    # 添加按钮来测试其他图表
    button_frame = ttk.Frame(root)
    button_frame.pack(fill=tk.X, pady=5)
    
    def show_error_dist():
        errors = np.random.normal(0, 1, 1000)
        draw_error_distribution(chart, errors)
    
    def show_conf_matrix():
        matrix = [[150, 20, 5], [15, 180, 10], [5, 10, 120]]
        draw_confusion_matrix(chart, matrix, labels=["A", "B", "C"])
    
    def show_training_time():
        phases = ["数据加载", "特征处理", "训练", "评估", "保存模型"]
        times = [5.2, 15.7, 45.3, 8.6, 3.2]
        draw_training_time(chart, phases, times)
    
    def show_alg_comparison():
        algs = {
            "随机森林": {"accuracy": 0.85, "training_time": 30},
            "梯度提升": {"accuracy": 0.88, "training_time": 45},
            "逻辑回归": {"accuracy": 0.75, "training_time": 15},
            "SVM": {"accuracy": 0.82, "training_time": 60}
        }
        draw_algorithm_comparison(chart, algs)
    
    ttk.Button(button_frame, text="误差分布", command=show_error_dist).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="混淆矩阵", command=show_conf_matrix).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="训练耗时", command=show_training_time).pack(side=tk.LEFT, padx=5)
    ttk.Button(button_frame, text="算法对比", command=show_alg_comparison).pack(side=tk.LEFT, padx=5)
    
    root.mainloop() 