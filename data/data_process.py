

import pandas as pd






class DataProcessor:
    def __init__(self, weather_file, gongdi_file):
        self.weather_file = weather_file
        self.gongdi_file = gongdi_file

    def read_data(self):
        # 读取天气数据
        self.weather_data = pd.read_csv(self.weather_file)
        print("read cvs success")
        self.weather_data['日期'] = pd.to_datetime(self.weather_data['日期'])

        # 读取工地数据
        self.gongdi_data = pd.read_excel(self.gongdi_file, sheet_name='Sheet1', engine='openpyxl')
        print("read xlsx success")
        self.gongdi_data['消纳场名称'] = self.gongdi_data['消纳场名称'].astype(str)  # 确保数据类型一致
        self.gongdi_data['消核时间'] = pd.to_datetime(self.gongdi_data['消核时间'])
        self.gongdi_data['使用时间'] = pd.to_datetime(self.gongdi_data['使用时间'])

    def process_data(self):
        # 统计每天的车型数量
        self.gongdi_data['日期'] = self.gongdi_data['消核时间'].dt.date
        
        # 统计每种车型的出现次数
        # daily_counts = self.gongdi_data.groupby(['日期', '车型']).size().unstack(fill_value=0)
        daily_counts = self.gongdi_data.groupby(['日期', '工地名称', '消纳场名称', '车型']).size().unstack(fill_value=0)

        # 为确保车型1、2、3都显示，即使没有出现
        for model in [1, 2, 3]:
            if model not in daily_counts.columns:
                daily_counts[model] = 0
        daily_counts = daily_counts[[1, 2, 3]]  # 确保顺序

        # 将日期转为DataFrame，并合并天气数据
        daily_counts.reset_index(inplace=True)
        daily_counts['日期'] = pd.to_datetime(daily_counts['日期'])

        # gongdi_aggregated = self.gongdi_data.groupby(['日期']).agg({
        #     '工地名称': 'first',  # 按日期取第一个工地名称
        #     '消纳场名称': 'first',
        # }).reset_index()
        
        # gongdi_aggregated = self.gongdi_data.groupby(['日期']).agg({
        #     '工地名称': lambda x: ', '.join(x.unique()),  # 合并同一天所有唯一工地名称
        #     '消纳场名称': lambda x: ', '.join(x.unique()),  # 合并同一天所有唯一消纳场名称
        # }).reset_index()

         # 将工地数据按日期、工地名称和车型展开，保留所有记录
        gongdi_aggregated = self.gongdi_data[['日期', '工地名称', '消纳场名称']].drop_duplicates()
        
        # 确保gongdi_aggregated中的日期列也是日期类型
        gongdi_aggregated['日期'] = pd.to_datetime(gongdi_aggregated['日期'])

        # 合并天气数据，确保包含所有天气数据的日期
        # merged_data = pd.merge(daily_counts, gongdi_aggregated, on='日期', how='left')
        merged_data = pd.merge(daily_counts, gongdi_aggregated, on=['日期', '工地名称', '消纳场名称'], how='left')

        # 使用右连接确保保留天气数据中的所有日期
        final_data = pd.merge(self.weather_data, merged_data, on='日期', how='left')

        return final_data

    def write_to_excel(self, output_file, data):
        # 将数据写入Excel文件
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Data')

    def run(self, output_file):
        self.read_data()
        processed_data = self.process_data()
        self.write_to_excel(output_file, processed_data)
        print("write success ...")



class DataProcessorXNC:
    def __init__(self, weather_file, xiaonachang_file):
        self.weather_file = weather_file
        self.xiaonachang_file = xiaonachang_file

    def read_data(self):
        # 读取天气数据
        self.weather_data = pd.read_csv(self.weather_file)
        print("read cvs success")
        self.weather_data['日期'] = pd.to_datetime(self.weather_data['日期'])

        # 读取消纳场数据
        self.xiaonachang_data = pd.read_excel(self.xiaonachang_file, sheet_name='Sheet1', engine='openpyxl')
        print("read xlsx success")
        self.xiaonachang_data['消纳场名称'] = self.xiaonachang_data['消纳场名称'].astype(str)  # 确保数据类型一致
        self.xiaonachang_data['工地名称'] = self.xiaonachang_data['site_name'].astype(str)
        self.xiaonachang_data['消核时间'] = pd.to_datetime(self.xiaonachang_data['消核时间'])
        self.xiaonachang_data['使用时间'] = pd.to_datetime(self.xiaonachang_data['使用时间'])

    def process_data(self):
        # 统计每天的车型数量
        self.xiaonachang_data['日期'] = self.xiaonachang_data['消核时间'].dt.date
        
        # 统计每种车型的出现次数
        # daily_counts = self.xiaonachang_data.groupby(['日期', '车型']).size().unstack(fill_value=0)
        daily_counts = self.xiaonachang_data.groupby(['日期', '消纳场名称', '车型']).size().unstack(fill_value=0)

        # 为确保车型1、2、3都显示，即使没有出现
        for model in [1, 2, 3]:
            if model not in daily_counts.columns:
                daily_counts[model] = 0
        daily_counts = daily_counts[[1, 2, 3]]  # 确保顺序

        # 将日期转为DataFrame，并合并天气数据
        daily_counts.reset_index(inplace=True)
        daily_counts['日期'] = pd.to_datetime(daily_counts['日期'])

        # gongdi_aggregated = self.xiaonachang_data.groupby(['日期']).agg({
        #     '工地名称': 'first',  # 按日期取第一个工地名称
        #     '消纳场名称': 'first',
        # }).reset_index()
        
        # gongdi_aggregated = self.xiaonachang_data.groupby(['日期']).agg({
        #     '工地名称': lambda x: ', '.join(x.unique()),  # 合并同一天所有唯一工地名称
        #     '消纳场名称': lambda x: ', '.join(x.unique()),  # 合并同一天所有唯一消纳场名称
        # }).reset_index()

         # 将工地数据按日期、工地名称和车型展开，保留所有记录
        xiaonachang_aggregated = self.xiaonachang_data[['日期', '工地名称', '消纳场名称']].drop_duplicates()
        
        # 确保gongdi_aggregated中的日期列也是日期类型
        xiaonachang_aggregated['日期'] = pd.to_datetime(xiaonachang_aggregated['日期'])

        # 合并天气数据，确保包含所有天气数据的日期
        # merged_data = pd.merge(daily_counts, xiaonachang_aggregated, on='日期', how='left')
        merged_data = pd.merge(daily_counts, xiaonachang_aggregated, on=['日期', '消纳场名称'], how='left')

        # 使用右连接确保保留天气数据中的所有日期
        final_data = pd.merge(self.weather_data, merged_data, on='日期', how='left')

        return final_data

    def write_to_excel(self, output_file, data):
        # 将数据写入Excel文件
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            data.to_excel(writer, index=False, sheet_name='Data')

    def run(self, output_file):
        self.read_data()
        processed_data = self.process_data()
        self.write_to_excel(output_file, processed_data)
        print("write success ...")
