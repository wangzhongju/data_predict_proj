

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


# # Define the device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TimeSeriesDataset(Dataset):
    def __init__(self, data, look_back):
        self.data = data
        self.look_back = look_back
        
    def __len__(self):
        return len(self.data) - self.look_back
    
    def __getitem__(self, idx):
        X = self.data[idx:idx+self.look_back]
        y = self.data[idx+self.look_back]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class DataPreprocessor:
    def __init__(self, look_back):
        self.look_back = look_back
        self.scaler = MinMaxScaler()
    
    def preprocess(self, data):
        data_np = data.values.reshape(-1, 1)  # Convert DataFrame to NumPy array
        data_normalized = self.scaler.fit_transform(data_np)
        dataset = TimeSeriesDataset(data_normalized, self.look_back)
        return dataset

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, 1))


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        # print("x: ", x.shape)
        # print("h0: ", h0.shape)
        # print("c0: ", c0.shape)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LSTMForecast:
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0.0, model_path=None):
        self.model = LSTMModel(input_size, hidden_size, output_size, num_layers, dropout)
        self.model_path = model_path
        self.data_preprocessor = None
        self.scaler = MinMaxScaler()
    
    def train(self, data, look_back, epochs=100, lr=0.01):
        self.data_preprocessor = DataPreprocessor(look_back)
        dataset = self.data_preprocessor.preprocess(data)
        train_loader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        self.model.train()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                # batch_X = batch_X.unsqueeze(-1)  # (batch_size, seq_len, input_size)
                optimizer.zero_grad()
                print("batch_X: " , batch_X)
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
    
    def predict(self, data, weeks=4, look_back=1):
        if self.data_preprocessor is None:
            self.data_preprocessor = DataPreprocessor(look_back)
        
        self.model.eval()
        predictions = []
        with torch.no_grad():
            # data_normalized = self.data_preprocessor.scaler.transform(data.reshape(-1, 1))
            data_np = data.values.reshape(-1, 1)  # Convert DataFrame to NumPy array
            data_normalized = self.data_preprocessor.scaler.transform(data_np)
            input_seq = torch.tensor(data_normalized[-look_back:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            
            for _ in range(weeks):
                pred = self.model(input_seq)
                predictions.append(pred.item())
                input_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(0).unsqueeze(1)), dim=1)
        
        return self.data_preprocessor.inverse_transform(np.array(predictions).reshape(-1, 1))
