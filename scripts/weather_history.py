

import requests
import pandas as pd
from datetime import datetime

class WeatherDataFetcher:
    def __init__(self, api_key, lat, lon):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.base_url = "http://api.openweathermap.org/data/2.5/onecall/timemachine"

    def fetch_data(self, date):
        timestamp = int(datetime.strptime(date, "%Y-%m-%d").timestamp())
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'dt': timestamp,
            'appid': self.api_key,
            'units': 'metric' # 摄氏度
        }
        response = requests.get(self.base_url, params=params)
        print("res: ", response.json())
        return response.json()

    def parse_data(self, data):
        weather_data = {
            'date': [],
            'temperature': [],
            'humidity': [],
            'precipitation': []
        }
        for hourly_data in data.get('hourly', []):
            weather_data['date'].append(datetime.fromtimestamp(hourly_data['dt']).strftime('%Y-%m-%d %H:%M:%S'))
            weather_data['temperature'].append(hourly_data['temp'])
            weather_data['humidity'].append(hourly_data['humidity'])
            weather_data['precipitation'].append(hourly_data.get('rain', {}).get('1h', 0))
        return pd.DataFrame(weather_data)

    def get_weather_data(self, date):
        data = self.fetch_data(date)
        df = self.parse_data(data)
        return df



if __name__ == "__main__":
    API_KEY = 'c417d77fb57a1346a1ca21701e517c8f'
    LAT = '29.612034999999995'
    LON = '106.55116900000007'
    
    fetcher = WeatherDataFetcher(API_KEY, LAT, LON)
    date = '2024-08-12'
    weather_df = fetcher.get_weather_data(date)
    print(weather_df)