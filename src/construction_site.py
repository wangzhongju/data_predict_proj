

import pandas as pd
import sys
import os


from scripts.tools import calculate_remaining_capacity
from scripts.tools import predict_future_capacity_linear
from scripts.tools import predict_future_capacity_foreast
# from scripts.tools import predict_future_capacity_lstm
from scripts.tools import predict_future_capacity_arima


class ConstructionSite:
    # 初始化时接收工地名称和总消纳量
    def __init__(self, name, total_capacity, style, freq):
        self.name = name
        self.total_capacity = total_capacity
        self.color = style['color']
        self.linestyle = style['linestyle']
        self.freq = freq
        self.data = pd.DataFrame(columns=['消纳场名称', '消核时间', '使用时间', '车型', '消纳量', '消纳时间'])
    
    # 将数据添加到当前工地的数据框中
    def add_data(self, df):
        df_filtered = df[df['工地名称'] == self.name]
        self.data = pd.concat([self.data, df_filtered])
    
    # 计算每周消耗量和剩余消纳量
    def calculate_remaining_capacity(self):
        self.data_resampled = calculate_remaining_capacity(self.data, self.total_capacity, self.freq)
    
    # 预测
    def predict_future_capacity(self, config, periods=4):
        # 时间序列模型
        model_path = os.path.join(config.get('model_path'), '{}.pth'.format(self.name))
        self.data_resampled_future = predict_future_capacity_arima(self.data_resampled,
                                                                   config, model_path,
                                                                   periods,
                                                                   freq=self.freq)
        # 深度学习模型
        # model_path = os.path.join(config.get('model_path'), '{}.pth'.format(self.name))
        # self.data_resampled_future = predict_future_capacity_lstm(self.data_resampled, config, model_path, weeks)
        # self.data_resampled_future = predict_future_capacity_lstm(self.data_resampled, weeks)
        # 线性回归
        # self.data_resampled_future = predict_future_capacity_linear(self.data_resampled, weeks)
        # 随机森林
        # self.data_resampled_future = predict_future_capacity_foreast(self.data_resampled, weeks)
    
    # 绘制每个工地的剩余消纳量曲线
    def plot_remaining_capacity(self, ax):
        self.data_resampled['剩余消纳量'].plot(ax=ax,
                                            color=self.color,
                                            linestyle=self.linestyle,
                                            marker='o',
                                            label=self.name)
    
    # 绘制每个工地的累计消纳量曲线
    def plot_cumulative_capacity(self, ax):
        self.data_resampled['累计消耗'].plot(ax=ax,
                                            color=self.color,
                                            linestyle=self.linestyle,
                                            marker='o',
                                            label=self.name)
    
    # 绘制未来几周的累计消纳量曲线
    def plot_future_capacity(self, ax):
        self.data_resampled_future['剩余消纳量'].plot(ax=ax,
                                                   label=f'{self.name} (预测)',
                                                   linestyle='--',
                                                   color=self.color)