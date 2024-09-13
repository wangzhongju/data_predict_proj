

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.font_manager import FontProperties

from src.construction_site import ConstructionSite
from src.disposal_site import DisposalSite


construction_styles = dict({
        "高新区西永组团F分区F03地块项目": {"color": "blue", "linestyle": "-"},
        "高新区西永组团Z分区Z46地块项目": {"color": "green", "linestyle": "--"},
        "寨山坪生态居住小区三期": {"color": "red", "linestyle": "-."},
        "科学田园项目中梁山片区新店村示范区建渣清理工程": {"color": "purple", "linestyle": ":"},
        "西科四路工程": {"color": "orange", "linestyle": "-"},
        "曾家“科研港”片区一路网工程": {"color": "cyan", "linestyle": "--"}
})
disposal_styles = dict({
        "花都湖土地整治项目": {"color": "magenta", "linestyle": "-"},
        "玉龙二号消纳场": {"color": "brown", "linestyle": "--"},
        "仓储基地土地整治工程": {"color": "grey", "linestyle": "-."},
        "新州大道路基平场项目": {"color": "black", "linestyle": ":"},
        "巴福大健康片区整治": {"color": "pink", "linestyle": "-"},
        "胡家沟二号（1号口）": {"color": "lightblue", "linestyle": "--"}
})




class ConstructionSitesManager:
    # 初始化时从 Excel 文件读取数据
    def __init__(self, construction_file, disposal_file, config, sheet_name="Sheet1"):
        self.config = config
        self.font = FontProperties(fname=self.config.get('font_file'))
        self.construction_styles = construction_styles
        self.disposal_styles = disposal_styles
        self.construction_file = construction_file
        self.disposal_file = disposal_file
        self.sheet_name = sheet_name
        self.freq = self.config.get('freq', 'W')
        # self.df_construction = pd.read_excel(self.construction_file, sheet_name=self.sheet_name, engine='openpyxl')
        # self.df_disposal = pd.read_excel(self.disposal_file, sheet_name=self.sheet_name, engine='openpyxl')
        # self.sites_construction = {}
        # self.sites_disposal = {}
        # self.init_sites()

    def init_sites(self, construction_sites=None, disposal_sites=None):
        if construction_sites is None:
            construction_sites = self.config.get('construction_sites', {})
        if disposal_sites is None:
            disposal_sites = self.config.get('disposal_sites', {})
        
        self.sites_construction = {}
        self.sites_disposal = {}

        for name, capacity in construction_sites.items():
            style = self.construction_styles.get(name, {'color': 'black', 'linestyle': '-'})
            self.sites_construction[name] = ConstructionSite(name, capacity, style, self.freq)
        for name, capacity in disposal_sites.items():
            style = self.disposal_styles.get(name, {'color': 'black', 'linestyle': '-'})
            self.sites_disposal[name] = DisposalSite(name, capacity, style, self.freq)
    
    # 将数据添加到所有工地对象中
    def add_data_to_sites_construction(self):
        self.df_construction = pd.read_excel(self.construction_file, sheet_name=self.sheet_name, engine='openpyxl')
        for name, site in self.sites_construction.items():
            site.add_data(self.df_construction)
            print("{} add data success".format(name))

    # 将数据添加到所有消纳场对象中
    def add_data_to_sites_disposal(self):
        self.df_disposal = pd.read_excel(self.disposal_file, sheet_name=self.sheet_name, engine='openpyxl')
        for name, site in self.sites_disposal.items():
            site.add_data(self.df_disposal)
            print("{} add data success".format(name))
    
    # 计算所有工地的剩余消纳量
    def calculate_all_remaining_capacities_construction(self):
        for site in self.sites_construction.values():
            site.calculate_remaining_capacity()

    # 计算所有消纳场的剩余消纳量
    def calculate_all_remaining_capacities_disposal(self):
        for site in self.sites_disposal.values():
            site.calculate_remaining_capacity()
    
    # 预测所有工地的消耗量
    def forecast_all_sites_construction(self, periods=4):
        for site in self.sites_construction.values():
            site.predict_future_capacity(self.config, periods)
    
    # 预测所有消纳场的消耗量
    def forecast_all_sites_disposal(self, periods=4):
        for site in self.sites_disposal.values():
            site.predict_future_capacity(self.config, periods)
    
    # 绘制所有的剩余消纳量曲线
    def plot_all_sites_gd(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for site in self.sites_construction.values():
            site.plot_remaining_capacity(ax)
            # site.plot_cumulative_capacity(ax)
            site.plot_future_capacity(ax)
        ax.set_title('每个工地渣土每周累计消纳量曲线', fontproperties=self.font)
        ax.set_xlabel('日期', fontproperties=self.font)
        ax.set_ylabel('累计消纳量 (立方米)', fontproperties=self.font)
        ax.legend(prop=self.font)
    
    # 绘制所有的剩余消纳量曲线
    def plot_all_sites_xnc(self):
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        for site in self.sites_disposal.values():
            site.plot_remaining_capacity(ax1)
            # site.plot_cumulative_capacity(ax1)
            site.plot_future_capacity(ax1)
        ax1.set_title('每个消纳场渣土每周累计消纳量曲线', fontproperties=self.font)
        ax1.set_xlabel('日期', fontproperties=self.font)
        ax1.set_ylabel('累计消纳量 (立方米)', fontproperties=self.font)
        ax1.legend(prop=self.font)

    # 绘制所有的剩余消纳量曲线
    def plot_save(self):
        plt.savefig('res.png')
        plt.close('all')

    def plot_show(self):
        # 开启交互模式
        plt.ion()
        # 显示图像
        plt.show()
        # 等待用户输入，直到关闭图像窗口
        input("按 Enter 键退出显示...")
        # 关闭图像窗口
        plt.ioff()  # 关闭交互模式
        plt.close('all')  # 关闭所有打开的图像窗口

    def forecast_all_sites(self, items_name, items_dict, error_msg, periods=4, config=None):
        if config:
            self.config = config

        if not items_name:
            print(f"Please input {error_msg} ...")
            return False

        missing_items = [name for name in items_name if name not in items_dict]
        if missing_items:
            print(f"====error: {', '.join(missing_items)} not found in {error_msg}.")
            return False

        # 获取存在于 items_dict 中的项
        items = [value for key, value in items_dict.items() if key in items_name]
        for item in items:
            item.predict_future_capacity(self.config, periods)
        return True

    # 预测指定工地的消耗量
    def forecast_all_sites_construction_app(self, periods=4, construction_name=None, config=None):
        self.construction_items = [value for key, value in self.sites_construction.items() if key in construction_name]
        return self.forecast_all_sites(construction_name, self.sites_construction, 'sites_construction', periods, config)

    # 预测指定消纳场的消耗量
    def forecast_all_sites_disposal_app(self, periods=4, disposal_name=None, config=None):
        self.disposal_items = [value for key, value in self.sites_disposal.items() if key in disposal_name]
        return self.forecast_all_sites(disposal_name, self.sites_disposal, 'sites_disposal', periods, config)

    # 绘制所有的剩余消纳量曲线
    def plot_all_sites_gd_app(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        for site in self.sites_construction.values():
            if site in self.construction_items:
                site.plot_remaining_capacity(ax)
                # site.plot_cumulative_capacity(ax)
                site.plot_future_capacity(ax)
        ax.set_title('每个工地渣土每周累计消纳量曲线', fontproperties=self.font)
        ax.set_xlabel('日期', fontproperties=self.font)
        ax.set_ylabel('累计消纳量 (立方米)', fontproperties=self.font)
        ax.legend(prop=self.font)
    
    # 绘制所有的剩余消纳量曲线
    def plot_all_sites_xnc_app(self):
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        for site in self.sites_disposal.values():
            if site in self.disposal_items:
                site.plot_remaining_capacity(ax1)
                # site.plot_cumulative_capacity(ax1)
                site.plot_future_capacity(ax1)
        ax1.set_title('每个消纳场渣土每周累计消纳量曲线', fontproperties=self.font)
        ax1.set_xlabel('日期', fontproperties=self.font)
        ax1.set_ylabel('累计消纳量 (立方米)', fontproperties=self.font)
        ax1.legend(prop=self.font)
