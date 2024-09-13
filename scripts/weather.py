

import requests
import pandas as pd
from datetime import datetime

class WeatherDataFetcher:
    def __init__(self, api_key, lat, lon):
        self.api_key = api_key
        self.lat = lat
        self.lon = lon
        self.base_url = "http://api.openweathermap.org/data/2.5/weather"

    def fetch_data(self):
        params = {
            'lat': self.lat,
            'lon': self.lon,
            'appid': self.api_key,
            'units': 'metric'  # 摄氏度
        }
        response = requests.get(self.base_url, params=params)
        print("res: ", response.json())
        return response.json()

    def parse_data(self, data):
        weather_data = {
            'date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'temperature': [data['main']['temp']],
            'humidity': [data['main']['humidity']],
            'precipitation': [data.get('rain', {}).get('1h', 0)],
            'weather': [data['weather'][0]['description']]
        }
        return pd.DataFrame(weather_data)

    def get_weather_data(self):
        data = self.fetch_data()
        df = self.parse_data(data)
        return df

if __name__ == "__main__":
    API_KEY = 'c417d77fb57a1346a1ca21701e517c8f'
    LAT = '29.612034999999995'
    LON = '106.55116900000007'
    
    fetcher = WeatherDataFetcher(API_KEY, LAT, LON)
    weather_df = fetcher.get_weather_data()
    print(weather_df)