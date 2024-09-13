import threading
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.font_manager import FontProperties
# from fbprophet import Prophet  # python >= 3.8
from sklearn.linear_model import LinearRegression
import numpy as np


# # 配置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
# plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

font = FontProperties(fname='/home/cjy/wzj/github/Znzx/algorithm/config/simhei.ttf') 

def plot_disposal_curve(file_path, sheet_name='Sheet1'):
    # 从Excel文件中读取数据，使用 openpyxl 引擎
    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
    
    # 将时间列转换为 datetime 格式
    df['消核时间'] = pd.to_datetime(df['消核时间'])
    df['使用时间'] = pd.to_datetime(df['使用时间'])
    
    # 计算消纳量
    df['消纳量'] = df['车型'] * 16  # 每车容量16m³
    
    # 按周分组，计算每周消纳量总和
    df.set_index('消核时间', inplace=True)
    weekly_data = df.resample('W').sum()
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(weekly_data.index, weekly_data['消纳量'], marker='o', linestyle='-', color='b')
    plt.title('渣土消纳曲线', fontproperties=font)
    plt.xlabel('时间', fontproperties=font)
    plt.ylabel('消纳量(立方米)', fontproperties=font)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


class DisposalCurvePlotter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.weekly_data = None
        self.vehicle_capacity = {1: 16, 2: 22, 3: 16}
        self.sheet_name = 'Sheet1'
    
    def read_data(self):
        """读取 Excel 文件"""
        self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name, engine='openpyxl')
        # self.df['消核时间'] = pd.to_datetime(self.df['消核时间'], unit='D', origin='1899-12-30')
        # self.df['使用时间'] = pd.to_datetime(self.df['使用时间'], unit='D', origin='1899-12-30')
        self.df['消核时间'] = pd.to_datetime(self.df['消核时间'])
        self.df['使用时间'] = pd.to_datetime(self.df['使用时间'])
    
    def calculate_disposal(self):
        """计算消纳量"""
        self.df['消纳量'] = self.df.apply(
            lambda row: (row['使用时间'] - row['消核时间']).days * self.vehicle_capacity[row['车型']], axis=1
        )
    
    def process_weekly_data(self):
        """按周分组计算消纳量"""
        self.df.set_index('消核时间', inplace=True)
        weekly_data = self.df.groupby(['工地名称', pd.Grouper(freq='W')])['消纳量'].sum().unstack(level=0).fillna(0)
        self.weekly_data = weekly_data
    
    def plot_disposal_curve(self):
        """绘制消纳曲线图"""
        plt.figure(figsize=(12, 8))
        plt.title('每周消纳曲线图', fontproperties=font)
        
        for col in self.weekly_data.columns:
            plt.plot(self.weekly_data.index, self.weekly_data[col], label=col)
        
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xlabel('时间', fontproperties=font)
        plt.ylabel('消纳量', fontproperties=font)
        plt.legend(prop=font)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.savefig('disposal_curve.png')
        plt.show()

    def run(self):
        """运行所有步骤"""
        self.read_data()
        self.calculate_disposal()
        self.process_weekly_data()
        self.plot_disposal_curve()



class ConstructionSite:
    # 初始化时接收工地名称和总消纳量
    def __init__(self, name, total_capacity):
        self.name = name
        self.total_capacity = total_capacity
        self.data = pd.DataFrame(columns=['消纳场名称', '消核时间', '使用时间', '车型', '消纳量', '消纳时间'])
    
    # 将数据添加到当前工地的数据框中
    def add_data(self, df):
        df_filtered = df[df['工地名称'] == self.name]
        self.data = pd.concat([self.data, df_filtered])
    
    # 计算每周消耗量和剩余消纳量
    def calculate_remaining_capacity(self):
        # TODO: 不手动创建新列该行报错
        self.data['消纳量'] = self.data.apply(lambda row: self.calculate_volume(row['车型']), axis=1)
        # self.data['消纳时间'] = pd.to_datetime(self.data['消核时间'], unit='D', origin='1900-01-01')
        self.data['消纳时间'] = pd.to_datetime(self.data['消核时间'])
        self.data = self.data.sort_values('消纳时间')
        # Calculate weekly consumption
        self.data.set_index('消纳时间', inplace=True)
        self.data_resampled = self.data.resample('W').sum()
        self.data_resampled['每周消耗'] = self.data_resampled['消纳量']
        self.data_resampled['累计消耗'] = self.data_resampled['消纳量'].cumsum()
        self.data_resampled['剩余消纳量'] = self.total_capacity - self.data_resampled['累计消耗']

    # 根据车型计算每次的消纳量
    def calculate_volume(self, vehicle_type):
        volume_per_vehicle = {1: 16, 2: 22, 3: 16}
        if vehicle_type not in [1, 2, 3]:
            print("key error, vehicle_type must in [1, 2, 3]: ", vehicle_type)
        return volume_per_vehicle.get(vehicle_type, 0)

    # 线性回归预测
    def forecast_consumption_with_linear(self, periods=4):
        df_prophet = self.data_resampled[['消纳量']].reset_index()
        df_prophet['周数'] = np.arange(len(df_prophet))
        
        X = df_prophet[['周数']]
        y = df_prophet['消纳量']
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_weeks = np.arange(len(df_prophet), len(df_prophet) + periods).reshape(-1, 1)
        forecast_values = model.predict(future_weeks)
        
        future_dates = pd.date_range(start=self.data_resampled.index[-1] + pd.Timedelta(weeks=1), periods=periods, freq='W')
        self.forecast = pd.DataFrame({
            '消纳时间': future_dates,
            '预测消纳量': forecast_values
        })
    
    # 绘制每个工地的预测剩余消纳量曲线(linears)
    def plot_predict_remaining_capacity_linears(self, ax):
        self.data_resampled['剩余消纳量'].plot(ax=ax, label=self.name)
        if not self.forecast.empty:
            ax.plot(self.forecast['消纳时间'], self.forecast['预测消纳量'], 'r--', label='预测剩余消纳量')
    
    # 绘制每个工地的剩余消纳量曲线
    def plot_remaining_capacity(self, ax):
        self.data_resampled['剩余消纳量'].plot(ax=ax, label=self.name)
    
    # 绘制每个工地的累计消纳量曲线
    def plot_cumulative_capacity(self, ax):
        self.data_resampled['累计消耗'].plot(ax=ax, label=self.name)
        # self.data_resampled['每周消耗'].plot(ax=ax, label=self.name)



class ConstructionSitesManager:
    # 初始化时从 Excel 文件读取数据
    def __init__(self, file_path, sheet_name="Sheet1"):
        self.df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        self.sites = {}
        self.init_sites()
    
    # 定义了所有工地及其总消纳量
    def init_sites(self):
        # Define the sites with their total capacities
        sites_capacities = {
            "高新区西永组团F分区F03地块项目": 3000000,
            "高新区西永组团Z分区Z46地块项目": 1300000,
            "寨山坪生态居住小区三期": 3000000,
            "科学田园项目中梁山片区新店村示范区建渣清理工程": 2000000,
            "西科四路工程": 3000000,
            "曾家“科研港”片区一路网工程": 3000000
        }
        
        for name, capacity in sites_capacities.items():
            self.sites[name] = ConstructionSite(name, capacity)
    
    # 将数据添加到所有工地对象中
    def add_data_to_sites(self):
        for name, site in self.sites.items():
            site.add_data(self.df)
            print("{} add data success".format(name))
    
    # 计算所有工地的剩余消纳量
    def calculate_all_remaining_capacities(self):
        for site in self.sites.values():
            site.calculate_remaining_capacity()

    # 预测所有工地的消耗量
    def forecast_all_sites(self, periods=4):
        for site in self.sites.values():
            # site.forecast_consumption_with_prophet(periods)
            site.forecast_consumption_with_linear(periods)
    
    # 绘制所有工地的剩余消纳量曲线
    def plot_all_sites(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        for site in self.sites.values():
            site.plot_cumulative_capacity(ax)
            # site.plot_predict_remaining_capacity_linears(ax1)
        ax.set_title('每个工地渣土剩余消纳量曲线', fontproperties=font)
        ax.set_xlabel('日期', fontproperties=font)
        ax.set_ylabel('剩余消纳量 (立方米)', fontproperties=font)
        ax.legend(prop=font)

        ax1.set_title('每个工地渣土剩余消纳量曲线', fontproperties=font)
        ax1.set_xlabel('日期', fontproperties=font)
        ax1.set_ylabel('剩余消纳量 (立方米)', fontproperties=font)
        ax1.legend(prop=font)

        # plt.savefig('meizhouxiaohao.png')
        plt.show()




if __name__ == "__main__":
    # file_path = '/home/cjy/wzj/github/Znzx/algorithm/data/gongdi.xlsx'
    # plot_disposal_curve(file_path)

    # file_path = '/home/cjy/wzj/github/Znzx/algorithm/data/gongdi.xlsx'  # 替换为你的文件路径
    # plotter = DisposalCurvePlotter(file_path)
    # plotter.run()

    manager = ConstructionSitesManager("/home/cjy/wzj/github/Znzx/algorithm/data/gongdi.xlsx")
    manager.add_data_to_sites()
    manager.calculate_all_remaining_capacities()
    manager.forecast_all_sites(periods=8)  # 预测未来8周
    manager.plot_all_sites()
