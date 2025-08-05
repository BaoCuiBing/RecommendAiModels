"""
性能监控页面模块
提供模型训练性能指标可视化和监控功能
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog
import logging
import json
import threading
import time
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib
from matplotlib import font_manager

# 导入基础页面类
from ui.pages.base_page import BasePage

# 导入自定义图表模块
import ui.charts.performance_charts as performance_charts

# 导入线程池工具
from utils.thread_pool import submit_task, get_task_status

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("monitor_page")

# 从train_page导入中文字体配置
from ui.pages.train_page import configure_matplotlib_chinese

# 配置中文字体
configure_matplotlib_chinese()

class MonitorPage(BasePage):
    """性能监控页面类"""
    
    def __init__(self, parent, controller=None, **kwargs):
        """
        初始化性能监控页面
        
        Args:
            parent: 父级窗口或Frame
            controller: 页面控制器
            **kwargs: 传递给BasePage的参数
        """
        # 监控相关变量
        self.is_monitoring = False
        self.monitoring_thread = None
        self.log_file_path = tk.StringVar(value="")
        self.refresh_interval = tk.IntVar(value=1000)  # 毫秒
        self.selected_metric = tk.StringVar(value="准确率")
        self.performance_data = []
        self.chart_type = tk.StringVar(value="折线图")
        
        super().__init__(parent, controller, **kwargs)
        logger.debug("性能监控页面已初始化")
    
    def _create_widgets(self):
        """创建页面小部件"""
        # 设置页面标题
        self.title_label = ttk.Label(
            self, 
            text="性能监控", 
            font=("Helvetica", 18, "bold")
        )
        self.title_label.pack(pady=(20, 10))
        
        # 创建主框架
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # 左侧：控制面板
        control_frame = ttk.LabelFrame(main_frame, text="监控控制", width=320)
        control_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10), pady=10, ipadx=10, ipady=5)
        control_frame.pack_propagate(False)  # 防止子组件影响frame大小
        
        # 日志文件路径
        log_frame = ttk.Frame(control_frame)
        log_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(log_frame, text="日志文件:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Entry(log_frame, textvariable=self.log_file_path, width=25).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(log_frame, text="浏览...", command=self._browse_log_file).pack(side=tk.LEFT, padx=(5, 0))
        
        # 刷新间隔
        interval_frame = ttk.Frame(control_frame)
        interval_frame.pack(fill=tk.X, padx=10, pady=10)
        
        ttk.Label(interval_frame, text="刷新间隔(ms):").pack(side=tk.LEFT, padx=(0, 5))
        interval_spinbox = ttk.Spinbox(
            interval_frame, 
            from_=100, 
            to=10000, 
            increment=100, 
            textvariable=self.refresh_interval,
            width=10
        )
        interval_spinbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # 监控指标选择
        metric_frame = ttk.LabelFrame(control_frame, text="监控指标")
        metric_frame.pack(fill=tk.X, padx=10, pady=10)
        
        metrics = ["准确率", "损失函数", "精确率", "召回率", "F1分数", "AUC", "训练时间"]
        
        for i, metric in enumerate(metrics):
            ttk.Radiobutton(
                metric_frame,
                text=metric,
                variable=self.selected_metric,
                value=metric,
                command=self._update_chart
            ).pack(anchor=tk.W, padx=10, pady=3)
        
        # 图表类型选择
        chart_frame = ttk.LabelFrame(control_frame, text="图表类型")
        chart_frame.pack(fill=tk.X, padx=10, pady=10)
        
        chart_types = ["折线图", "柱状图", "散点图", "热力图"]
        
        for chart_type in chart_types:
            ttk.Radiobutton(
                chart_frame,
                text=chart_type,
                variable=self.chart_type,
                value=chart_type,
                command=self._update_chart
            ).pack(anchor=tk.W, padx=10, pady=3)
        
        # 监控控制按钮
        button_frame = ttk.Frame(control_frame)
        button_frame.pack(fill=tk.X, padx=10, pady=15)
        
        self.start_button = ttk.Button(
            button_frame, 
            text="开始监控",
            command=self._start_monitoring,
            width=15
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame, 
            text="停止监控",
            command=self._stop_monitoring,
            width=15,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 导出数据按钮
        export_frame = ttk.Frame(control_frame)
        export_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(
            export_frame, 
            text="导出数据",
            command=self._export_data,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            export_frame, 
            text="导出图表",
            command=self._export_chart,
            width=15
        ).pack(side=tk.LEFT, padx=5)
        
        # 右侧：图表显示
        chart_container = ttk.LabelFrame(main_frame, text="性能图表")
        chart_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        
        # 图表切换标签页
        self.chart_notebook = ttk.Notebook(chart_container)
        self.chart_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 单指标图表页面
        self.single_chart_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.single_chart_frame, text="单指标")
        
        # 多指标对比页面
        self.multi_chart_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.multi_chart_frame, text="多指标对比")
        
        # 相关性分析页面
        self.correlation_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.correlation_frame, text="相关性分析")
        
        # 数据表格页面
        self.data_frame = ttk.Frame(self.chart_notebook)
        self.chart_notebook.add(self.data_frame, text="数据表格")
        
        # 创建单指标图表
        self.figure1 = Figure(figsize=(7, 4.5), dpi=100)
        self.figure1.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)
        self.plot1 = self.figure1.add_subplot(111)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, self.single_chart_frame)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建多指标对比图表
        self.figure2 = Figure(figsize=(7, 4.5), dpi=100)
        self.figure2.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)
        self.plot2 = self.figure2.add_subplot(111)
        self.canvas2 = FigureCanvasTkAgg(self.figure2, self.multi_chart_frame)
        self.canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建相关性图表
        self.figure3 = Figure(figsize=(7, 4.5), dpi=100)
        self.figure3.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)
        self.plot3 = self.figure3.add_subplot(111)
        self.canvas3 = FigureCanvasTkAgg(self.figure3, self.correlation_frame)
        self.canvas3.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 创建数据表格
        self.data_table_frame = ttk.Frame(self.data_frame)
        self.data_table_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 表格列定义
        self.table_columns = ("timestamp", "accuracy", "loss", "precision", "recall", "f1", "auc", "time")
        self.data_table = ttk.Treeview(
            self.data_table_frame, 
            columns=self.table_columns, 
            show="headings",
            height=20
        )
        
        # 设置列标题
        self.data_table.heading("timestamp", text="时间戳")
        self.data_table.heading("accuracy", text="准确率")
        self.data_table.heading("loss", text="损失")
        self.data_table.heading("precision", text="精确率")
        self.data_table.heading("recall", text="召回率")
        self.data_table.heading("f1", text="F1")
        self.data_table.heading("auc", text="AUC")
        self.data_table.heading("time", text="训练时间(秒)")
        
        # 设置列宽
        for col in self.table_columns:
            self.data_table.column(col, width=80)
        
        # 添加滚动条
        table_scroll = ttk.Scrollbar(self.data_table_frame, orient="vertical", command=self.data_table.yview)
        self.data_table.configure(yscrollcommand=table_scroll.set)
        
        # 布局表格和滚动条
        self.data_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        table_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 底部按钮
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=20, pady=15)
        
        ttk.Button(
            bottom_frame,
            text="返回",
            command=lambda: self.controller.show_page("TrainPage"),
            width=15
        ).pack(side=tk.RIGHT, padx=5)
        
        # 初始化图表
        self._init_charts()
    
    def _init_charts(self):
        """初始化所有图表"""
        # 初始化单指标图表
        self.plot1.clear()
        self.plot1.set_title("性能指标监控")
        self.plot1.set_xlabel("迭代次数")
        self.plot1.set_ylabel(self.selected_metric.get())
        self.plot1.grid(True, linestyle='--', alpha=0.7)
        self.canvas1.draw()
        
        # 初始化多指标对比图表
        self.plot2.clear()
        self.plot2.set_title("多指标性能对比")
        self.plot2.set_xlabel("指标")
        self.plot2.set_ylabel("值")
        self.plot2.grid(True, linestyle='--', alpha=0.7)
        self.canvas2.draw()
        
        # 初始化相关性图表
        self.plot3.clear()
        self.plot3.set_title("指标相关性分析")
        self.plot3.set_xlabel("")
        self.plot3.set_ylabel("")
        self.canvas3.draw()
    
    def _browse_log_file(self):
        """浏览选择日志文件"""
        file_path = filedialog.askopenfilename(
            title="选择日志文件",
            filetypes=[("日志文件", "*.log *.txt"), ("JSON文件", "*.json"), ("所有文件", "*.*")]
        )
        if file_path:
            self.log_file_path.set(file_path)
    
    def _start_monitoring(self):
        """开始监控"""
        # 参数验证
        if not self.log_file_path.get():
            self.show_message("请选择日志文件", "warning")
            return
        
        if not os.path.exists(self.log_file_path.get()):
            self.show_message("日志文件不存在", "error")
            return
        
        # 更新UI状态
        self.is_monitoring = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # 清空数据
        self.performance_data = []
        
        # 清空表格
        for item in self.data_table.get_children():
            self.data_table.delete(item)
        
        # 初始化图表
        self._init_charts()
        
        # 启动监控线程
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_thread,
            daemon=True
        )
        self.monitoring_thread.start()
        
        logger.info(f"开始监控性能: {self.log_file_path.get()}")
    
    def _stop_monitoring(self):
        """停止监控"""
        if self.is_monitoring:
            self.is_monitoring = False
            
            # 更新UI状态
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            logger.info("停止监控性能")
    
    def _monitoring_thread(self):
        """监控线程函数"""
        try:
            # 模拟从日志文件读取数据
            # 实际项目中应该从真实日志文件解析数据
            self._generate_mock_data()
            
            # 监控循环
            while self.is_monitoring and self.winfo_exists():
                # 刷新图表
                self._update_chart()
                
                # 等待下一次刷新
                interval = self.refresh_interval.get() / 1000.0  # 转换为秒
                time.sleep(interval)
                
                # 模拟新数据
                if len(self.performance_data) < 50:  # 限制模拟数据数量
                    self._add_mock_data_point()
        
        except Exception as e:
            logger.error(f"监控过程出错: {str(e)}")
            if self.winfo_exists():
                self.show_message(f"监控过程出错: {str(e)}", "error")
        
        finally:
            # 恢复UI状态
            if self.winfo_exists():
                self.is_monitoring = False
                self.start_button.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED)
    
    def _generate_mock_data(self):
        """生成模拟数据"""
        # 清空现有数据
        self.performance_data = []
        
        # 生成初始模拟数据
        base_time = time.time()
        for i in range(10):
            # 随机生成各项指标数据
            data_point = {
                "timestamp": time.strftime("%H:%M:%S", time.localtime(base_time + i*60)),
                "iteration": i + 1,
                "accuracy": 0.5 + 0.04 * i + np.random.normal(0, 0.01),
                "loss": 1.0 - 0.08 * i + np.random.normal(0, 0.02),
                "precision": 0.6 + 0.03 * i + np.random.normal(0, 0.01),
                "recall": 0.55 + 0.035 * i + np.random.normal(0, 0.015),
                "f1": 0.57 + 0.032 * i + np.random.normal(0, 0.012),
                "auc": 0.62 + 0.03 * i + np.random.normal(0, 0.01),
                "time": 10 + i * 0.5 + np.random.normal(0, 0.5)
            }
            
            # 添加到数据列表
            self.performance_data.append(data_point)
            
            # 添加到表格
            self._add_data_to_table(data_point)
    
    def _add_mock_data_point(self):
        """添加一个模拟数据点"""
        if not self.performance_data:
            return
            
        last_point = self.performance_data[-1]
        base_time = time.time()
        iteration = last_point["iteration"] + 1
        
        # 创建新数据点
        new_point = {
            "timestamp": time.strftime("%H:%M:%S", time.localtime(base_time)),
            "iteration": iteration,
            "accuracy": min(0.95, last_point["accuracy"] + np.random.normal(0.01, 0.005)),
            "loss": max(0.05, last_point["loss"] - np.random.normal(0.01, 0.005)),
            "precision": min(0.95, last_point["precision"] + np.random.normal(0.008, 0.004)),
            "recall": min(0.95, last_point["recall"] + np.random.normal(0.007, 0.004)),
            "f1": min(0.95, last_point["f1"] + np.random.normal(0.007, 0.003)),
            "auc": min(0.97, last_point["auc"] + np.random.normal(0.006, 0.003)),
            "time": last_point["time"] + np.random.normal(0.5, 0.1)
        }
        
        # 添加到数据列表
        self.performance_data.append(new_point)
        
        # 添加到表格
        self._add_data_to_table(new_point)
    
    def _add_data_to_table(self, data_point):
        """添加数据到表格"""
        if not self.winfo_exists():
            return
            
        # 插入数据到表格
        self.data_table.insert("", tk.END, values=(
            data_point["timestamp"],
            f"{data_point['accuracy']:.4f}",
            f"{data_point['loss']:.4f}",
            f"{data_point['precision']:.4f}",
            f"{data_point['recall']:.4f}",
            f"{data_point['f1']:.4f}",
            f"{data_point['auc']:.4f}",
            f"{data_point['time']:.2f}"
        ))
        
        # 确保最新数据可见
        self.data_table.yview_moveto(1.0)
    
    def _update_chart(self):
        """更新图表显示"""
        if not self.performance_data or not self.winfo_exists():
            return
        
        # 获取当前选中的指标和图表类型
        metric = self.selected_metric.get()
        chart_type = self.chart_type.get()
        
        # 更新单指标图表
        self._update_single_chart(metric, chart_type)
        
        # 更新多指标对比图表
        self._update_multi_chart()
        
        # 更新相关性图表
        self._update_correlation_chart()
    
    def _update_single_chart(self, metric, chart_type):
        """
        更新单指标图表
        
        Args:
            metric: 选中的指标
            chart_type: 图表类型
        """
        # 提取数据
        iterations = [point["iteration"] for point in self.performance_data]
        
        # 映射指标名称到数据键
        metric_map = {
            "准确率": "accuracy",
            "损失函数": "loss",
            "精确率": "precision",
            "召回率": "recall",
            "F1分数": "f1",
            "AUC": "auc",
            "训练时间": "time"
        }
        
        metric_key = metric_map.get(metric, "accuracy")
        values = [point[metric_key] for point in self.performance_data]
        
        # 清除原图
        self.plot1.clear()
        
        # 绘制新图表
        if chart_type == "折线图":
            self.plot1.plot(iterations, values, 'b-', linewidth=1.5, marker='o', markersize=4)
        elif chart_type == "柱状图":
            self.plot1.bar(iterations, values, color='skyblue', alpha=0.7)
        elif chart_type == "散点图":
            self.plot1.scatter(iterations, values, color='darkblue', alpha=0.7)
        elif chart_type == "热力图":
            # 热力图需要二维数据，这里简单处理一下
            data_2d = np.array([values])
            im = self.plot1.imshow(data_2d, cmap='hot', aspect='auto')
            self.figure1.colorbar(im, ax=self.plot1)
        
        # 设置图表属性
        self.plot1.set_title(f"{metric}监控")
        self.plot1.set_xlabel("迭代次数")
        self.plot1.set_ylabel(metric)
        self.plot1.grid(True, linestyle='--', alpha=0.7)
        
        # 重新绘制
        self.canvas1.draw()
    
    def _update_multi_chart(self):
        """更新多指标对比图表"""
        if len(self.performance_data) == 0:
            return
        
        # 获取最新数据点
        latest_data = self.performance_data[-1]
        
        # 提取指标数据
        metrics = ["准确率", "精确率", "召回率", "F1分数", "AUC"]
        metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
        values = [latest_data[key] for key in metric_keys]
        
        # 清除原图
        self.plot2.clear()
        
        # 绘制新柱状图
        bars = self.plot2.bar(metrics, values, color='skyblue', alpha=0.7)
        
        # 在柱子上显示数值
        for bar, value in zip(bars, values):
            height = bar.get_height()
            self.plot2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', rotation=0, fontsize=8)
        
        # 设置图表属性
        self.plot2.set_title("最新指标对比")
        self.plot2.set_ylabel("值")
        self.plot2.set_ylim(0, 1.0)
        self.plot2.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 重新绘制
        self.canvas2.draw()
    
    def _update_correlation_chart(self):
        """更新相关性图表"""
        if len(self.performance_data) < 5:  # 需要足够的数据点
            return
        
        # 创建数据框
        df = pd.DataFrame(self.performance_data)
        
        # 选择要分析的列
        columns = ["accuracy", "precision", "recall", "f1", "auc"]
        corr_df = df[columns].corr()
        
        # 清除原图
        self.plot3.clear()
        
        # 绘制热力图
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=.5, ax=self.plot3, fmt='.2f', cbar_kws={"shrink": .8})
        
        # 设置图表属性
        self.plot3.set_title("指标相关性分析")
        
        # 重新绘制
        self.canvas3.draw()
    
    def _export_data(self):
        """导出监控数据"""
        if not self.performance_data:
            self.show_message("没有数据可导出", "warning")
            return
            
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="导出数据",
            defaultextension=".csv",
            filetypes=[("CSV文件", "*.csv"), ("Excel文件", "*.xlsx"), ("JSON文件", "*.json")]
        )
        
        if not file_path:
            return
            
        try:
            # 创建DataFrame
            df = pd.DataFrame(self.performance_data)
            
            # 根据文件扩展名选择导出格式
            if file_path.endswith(".csv"):
                df.to_csv(file_path, index=False)
            elif file_path.endswith(".xlsx"):
                df.to_excel(file_path, index=False)
            elif file_path.endswith(".json"):
                df.to_json(file_path, orient="records")
                
            self.show_message(f"数据已成功导出到: {file_path}", "info")
            
        except Exception as e:
            logger.error(f"导出数据出错: {str(e)}")
            self.show_message(f"导出数据出错: {str(e)}", "error")
    
    def _export_chart(self):
        """导出图表"""
        if not self.performance_data:
            self.show_message("没有数据可导出", "warning")
            return
            
        # 选择保存路径
        file_path = filedialog.asksaveasfilename(
            title="导出图表",
            defaultextension=".png",
            filetypes=[("PNG文件", "*.png"), ("PDF文件", "*.pdf"), ("SVG文件", "*.svg")]
        )
        
        if not file_path:
            return
            
        try:
            # 获取当前选中的选项卡索引
            current_tab = self.chart_notebook.index(self.chart_notebook.select())
            
            # 根据选中的选项卡选择要导出的图表
            if current_tab == 0:  # 单指标
                self.figure1.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 1:  # 多指标对比
                self.figure2.savefig(file_path, dpi=300, bbox_inches='tight')
            elif current_tab == 2:  # 相关性分析
                self.figure3.savefig(file_path, dpi=300, bbox_inches='tight')
                
            self.show_message(f"图表已成功导出到: {file_path}", "info")
            
        except Exception as e:
            logger.error(f"导出图表出错: {str(e)}")
            self.show_message(f"导出图表出错: {str(e)}", "error")
    
    def refresh(self):
        """刷新页面内容"""
        pass
    
    def on_show(self):
        """页面显示时的回调"""
        logger.debug("性能监控页面显示")
    
    def on_hide(self):
        """页面隐藏时的回调"""
        # 停止监控
        if self.is_monitoring:
            self._stop_monitoring()


# 测试代码
if __name__ == "__main__":
    root = tk.Tk()
    root.title("性能监控页面测试")
    root.geometry("1200x700")
    
    # 模拟控制器
    class MockController:
        def __init__(self):
            self.shared_data = {}
        
        def show_page(self, page_name):
            print(f"切换到页面: {page_name}")
    
    # 创建页面
    page = MonitorPage(root, MockController())
    page.show()
    
    # 显示窗口
    root.mainloop() 