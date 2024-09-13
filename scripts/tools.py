

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# import torch
# import torch.nn as nn
# import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# # Define the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=3, dropout=0.2):
#         super(LSTMModel, self).__init__()
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
#         self.fc = nn.Linear(hidden_size, output_size)
#         self.num_layers = num_layers
#         self.hidden_size = hidden_size

#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out

# def prepare_data(data, look_back=1):
#     data_values = data.values
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data_values)
    
#     X, y = [], []
#     for i in range(len(scaled_data) - look_back):
#         X.append(scaled_data[i:(i + look_back), 0])
#         y.append(scaled_data[i + look_back, 0])
    
#     X, y = np.array(X), np.array(y)
#     X = X.reshape((X.shape[0], X.shape[1], 1))
#     return X, y, scaler

# def predict_future_capacity_lstm(data_resampled, weeks=4, look_back=1, train_model=True, model_path='lstm_model.pth'):
#     if len(data_resampled) < look_back:
#         print("数据不足以进行预测")
#         return
    
#     # 准备数据
#     X, y, scaler = prepare_data(data_resampled[['剩余消纳量']], look_back)
#     X_train = torch.tensor(X, dtype=torch.float32).to(device)
#     y_train = torch.tensor(y, dtype=torch.float32).to(device)
    
#     # 创建LSTM模型
#     model = LSTMModel(input_size=1, hidden_size=50, output_size=1, num_layers=3, dropout=0.2).to(device)
    
#     if train_model:
#         # 训练模型
#         criterion = nn.MSELoss()
#         optimizer = optim.Adam(model.parameters(), lr=0.001)
        
#         model.train()
#         for epoch in range(1000):
#             optimizer.zero_grad()
#             output = model(X_train)
#             loss = criterion(output.squeeze(), y_train)
#             loss.backward()
#             optimizer.step()
#             if (epoch + 1) % 50 == 0:
#                 print(f'Epoch [{epoch + 1}/500], Loss: {loss.item():.4f}')
        
#         # 保存模型
#         torch.save(model.state_dict(), model_path)
#         print(f"模型已保存到 {model_path}")
#     else:
#         # 加载已保存的模型
#         if os.path.exists(model_path):
#             model.load_state_dict(torch.load(model_path))
#             model.eval()
#             print(f"模型已加载自 {model_path}")
#         else:
#             print(f"模型文件 {model_path} 不存在")
#             return
    
#     # 进行预测
#     model.eval()
#     last_data = data_resampled[['剩余消纳量']].values[-look_back:]
#     last_data_scaled = scaler.transform(last_data).reshape((1, look_back, 1))
#     last_data_tensor = torch.tensor(last_data_scaled, dtype=torch.float32).to(device)
    
#     future_predictions = []
#     for _ in range(weeks):
#         with torch.no_grad():
#             prediction = model(last_data_tensor).cpu().numpy()[0, 0]
#             future_predictions.append(prediction)
            
#             # 更新数据
#             new_data = np.array([[prediction]])
#             new_data_scaled = scaler.transform(new_data).reshape((1, look_back, 1))
#             last_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(device)
    
#     future_dates = pd.date_range(start=data_resampled.index[-1] + pd.DateOffset(weeks=1), periods=weeks, freq='W')
#     future_cumulative_consumption = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    
#     future_data = pd.DataFrame(index=future_dates, data={'剩余消纳量': future_cumulative_consumption.flatten()})
#     return future_data



from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import joblib


def preprocess_data(data, column_name):
    """
    过滤掉指定列中值为0的行。

    Parameters:
    - data: DataFrame，原始数据
    - column_name: str，指定需要过滤的列名

    Returns:
    - DataFrame，过滤后的数据
    """
    return data[data[column_name] != 0]


def make_data(data_resampled, y, periods, freq):
    """
    生成未来的假数据。

    Parameters:
    - data_resampled: DataFrame，原始数据，通常用于提取最后一个日期
    - y: Series，历史数据
    - periods: int，生成的未来数据的时间段数（周数或天数）
    - freq: str，生成数据的频率，'W' 表示按周，'D' 表示按天

    Returns:
    - DataFrame，包含未来的假数据
    """
    # 生成假数据
    last_value = y.iloc[-1] if len(y) > 0 else 0
    fake_forecast = np.full(periods, last_value)  # 用最后一个值填充预测数据
    
    # 生成未来日期
    if freq == 'D':
        future_dates = pd.date_range(start=data_resampled.index[-1] + pd.DateOffset(days=1), 
                                     periods=periods, freq='D')
    elif freq == 'W':
        future_dates = pd.date_range(start=data_resampled.index[-1] + pd.DateOffset(weeks=1), 
                                     periods=periods, freq='W')
    future_data = pd.DataFrame(index=future_dates, data={'剩余消纳量': fake_forecast})
    return future_data


def predict_future_capacity_arima(data_resampled, config, model_path, periods=4, freq='W'):
    """使用 ARIMA 模型预测未来几周的累计消纳量"""
    # 确保数据索引为日期时间类型
    if not isinstance(data_resampled.index, pd.DatetimeIndex):
        data_resampled.index = pd.to_datetime(data_resampled.index)
    
    data_resampled = preprocess_data(data_resampled, '每周消耗' if freq == 'W' else '每日消耗')
    y = data_resampled['剩余消纳量']

    if len(y) < 4:
        print("数据不足以进行预测, 返回假数据")
        future_data = make_data(data_resampled, y, periods, freq=freq)
        return future_data

    if config.get('train_model', True):
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # 自动选择 ARIMA 参数
                model = auto_arima(y, seasonal=False, stepwise=True)
            # 保存模型
            joblib.dump(model, model_path)
            print("save model: ", model_path)
        except Exception as e:
            print(f"模型训练失败: {e}")
            future_data = make_data(data_resampled, y, periods, freq=freq)
            return future_data
    else:
        # 从文件加载模型
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("load model: ", model_path)
        else:
            print("模型文件不存在，请训练模型并保存到指定路径。")
            future_data = make_data(data_resampled, y, periods, freq=freq)
            return future_data
    
    try:
        forecast = model.predict(n_periods=periods)
        forecast = np.array(forecast).flatten()
        max_value = y.max()
        forecast = np.clip(forecast, 0, max_value)

        if freq == 'D':
            future_dates = pd.date_range(start=data_resampled.index[-1] + pd.DateOffset(days=1), 
                                         periods=periods, freq='D')
        elif freq == 'W':
            future_dates = pd.date_range(start=data_resampled.index[-1] + pd.DateOffset(weeks=1), 
                                         periods=periods, freq='W')
        else:
            raise ValueError(f"不支持的频率: {freq}")

        future_data = pd.DataFrame(index=future_dates, data={'剩余消纳量': forecast})
        future_data.index = pd.DatetimeIndex(future_data.index)  # 确保索引为日期时间类型
        print("未来数据: ", future_data)
        return future_data

    except Exception as e:
        print(f"预测失败: {e}")
        future_data = make_data(data_resampled, y, periods, freq=freq)
        return future_data




def predict_future_capacity_linear(data_resampled, weeks=4):
    """预测未来几周的累计消纳量"""
    if len(data_resampled) < 2:
        print(f"数据不足以进行预测")
        return
    
    # 创建线性回归模型进行预测
    X = np.arange(len(data_resampled)).reshape(-1, 1)
    # y = data_resampled['累计消耗'].values
    y = data_resampled['剩余消纳量'].values
    
    model = LinearRegression()
    model.fit(X, y)
    
    # 预测未来几周的消耗量
    future_weeks = np.arange(len(data_resampled), len(data_resampled) + weeks).reshape(-1, 1)
    future_cumulative_consumption = model.predict(future_weeks)
    
    # 生成未来几周的日期索引
    future_dates = pd.date_range(start=data_resampled.index[-1] + pd.DateOffset(weeks=1), periods=weeks, freq='W')
    future_data = pd.DataFrame(index=future_dates, data={'剩余消纳量': future_cumulative_consumption})
    return future_data


def predict_future_capacity_foreast(data_resampled, weeks=4):
    """预测未来几周的累计消纳量"""
    if len(data_resampled) < 2:
        print("数据不足以进行预测")
        return
    
    # 准备特征和目标变量
    X = np.arange(len(data_resampled)).reshape(-1, 1)
    y = data_resampled['剩余消纳量'].values
    
    # 创建训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建随机森林回归模型进行训练
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 在测试集上评估模型
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"测试集均方误差: {mse:.2f}")
    
    # 预测未来几周的消耗量
    future_weeks = np.arange(len(data_resampled), len(data_resampled) + weeks).reshape(-1, 1)
    future_cumulative_consumption = model.predict(future_weeks)
    
    # 生成未来几周的日期索引
    future_dates = pd.date_range(start=data_resampled.index[-1] + pd.DateOffset(weeks=1), periods=weeks, freq='W')
    future_data = pd.DataFrame(index=future_dates, data={'剩余消纳量': future_cumulative_consumption})
    
    return future_data


def calculate_remaining_capacity(data, total_capacity, freq='W'):
    """
    计算消耗量和剩余消纳量。

    :param data: DataFrame, 包含 '车型' 和 '消纳时间' 列的数据
    :param total_capacity: int, 总容量
    :param freq: str, 重采样频率 ('W' 为每周, 'D' 为每日)
    :return: DataFrame, 包含 '每周消耗' 或 '每日消耗', '累计消耗', '剩余消纳量'
    """
    # # 检查并处理缺失值
    # if data['消纳时间'].isnull().any():
    #     raise ValueError("数据中包含缺失的消纳时间")
    
    # 计算消纳量
    data['消纳量'] = data.apply(lambda row: calculate_volume(row['车型']), axis=1)
    data['消纳时间'] = pd.to_datetime(data['消核时间'])
    # 排序和设置索引
    data.sort_values('消纳时间', inplace=True)
    data.set_index('消纳时间', inplace=True)
    # 重采样
    data_resampled = data.resample(freq).sum()
    # 计算消耗量
    # data_resampled['消耗量'] = data_resampled['消纳量']
    if freq == 'D':
        data_resampled['每日消耗'] = data_resampled['消纳量']
    elif freq == 'W':
        data_resampled['每周消耗'] = data_resampled['消纳量']
    # 计算累计消耗量
    data_resampled['累计消耗'] = data_resampled['消纳量'].cumsum()
    # 计算剩余消纳量
    data_resampled['剩余消纳量'] = total_capacity - data_resampled['累计消耗']
    return data_resampled


# 根据车型计算每次的消纳量
def calculate_volume(vehicle_type):
    volume_per_vehicle = {1: 16, 2: 22, 3: 16}
    if vehicle_type not in [1, 2, 3]:
        print("key error, vehicle_type must in [1, 2, 3]: ", vehicle_type)
    return volume_per_vehicle.get(vehicle_type, 0)




if __name__ == "__main__":
    data_resampled = pd.DataFrame({'剩余消纳量': [100, 90, 80, 70, 60]}, index=pd.date_range(start='2023-01-01', periods=5, freq='W'))
    future_data = predict_future_capacity_lstm(data_resampled)
    print(future_data)