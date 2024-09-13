
# -*- coding: utf-8 -*-
# @Date     : 2024/08/23
# @Author   : WZJ
# @File     : get_weather.py

import requests
from bs4 import BeautifulSoup  # pip install beautifulsoup4
import pandas as pd
import re

class WeatherScraper:
    """
    获取历史天气
    TODO: USER_AGENT 与 COOKIE nouse
    USER_AGENT = 'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'
    COOKIE = 'lianjia_uuid=9d3277d3-58e4-440e-bade-5069cb5203a4; UM_distinctid=16ba37f7160390-05f17711c11c3e-\
        454c0b2b-100200-16ba37f716618b; _smt_uid=5d176c66.5119839a; sensorsdata2015jssdkcross=%7B%22distinct_id%22%3A%2216b\
        a37f7a942a6-0671dfdde0398a-454c0b2b-1049088-16ba37f7a95409%22%2C%22%24device_id%22%3A%2216ba37f7a942a6-\
        0671dfdde0398a-454c0b2b-1049088-16ba37f7a95409%22%2C%22props%22%3A%7B%22%24latest_traffic_source_\
        type%22%3A%22%E7%9B%B4%E6%8E%A5%E6%B5%81%E9%87%8F%22%2C%22%24latest_referrer%22%3A%22%22%2C%22%24latest_\
        referrer_host%22%3A%22%22%2C%22%24latest_search_keyword%22%3A%22%E6%9C%AA%E5%8F%96%E5%88%B0%E5%80%BC_%E7%9\
        B%B4%E6%8E%A5%E6%89%93%E5%BC%80%22%7D%7D; _ga=GA1.2.1772719071.1561816174; Hm_lvt_9152f8221cb6243a53c83b9568\
        42be8a=1561822858; _jzqa=1.2532744094467475000.1561816167.1561822858.1561870561.3; CNZZDATA1253477573=987273979-\
        1561811144-%7C1561865554; CNZZDATA1254525948=879163647-1561815364-%7C1561869382; CNZZDATA1255633284=1986996647-\
        1561812900-%7C1561866923; CNZZDATA1255604082=891570058-1561813905-%7C1561866148; _qzja=1.1577983579.1561816168942.\
        1561822857520.1561870561449.1561870561449.1561870847908.0.0.0.7.3; select_city=110000; lianjia_ssid=4e1fa281-1ebf-\
        e1c1-ac56-32b3ec83f7ca; srcid=eyJ0Ijoie1wiZGF0YVwiOlwiMzQ2MDU5ZTQ0OWY4N2RiOTE4NjQ5YmQ0ZGRlMDAyZmFhODZmNjI1ZDQyNWU0O\
        GQ3MjE3Yzk5NzFiYTY4ODM4ZThiZDNhZjliNGU4ODM4M2M3ODZhNDNiNjM1NzMzNjQ4ODY3MWVhMWFmNzFjMDVmMDY4NWMyMTM3MjIxYjBmYzhkYWE1Mz\
        IyNzFlOGMyOWFiYmQwZjBjYjcyNmIwOWEwYTNlMTY2MDI1NjkyOTBkNjQ1ZDkwNGM5ZDhkYTIyODU0ZmQzZjhjODhlNGQ1NGRkZTA0ZTBlZDFiNmIxO\
        TE2YmU1NTIxNzhhMGQ3Yzk0ZjQ4NDBlZWI0YjlhYzFiYmJlZjJlNDQ5MDdlNzcxMzAwMmM1ODBlZDJkNmIwZmY0NDAwYmQxNjNjZDlhNmJkNDk3NGMzO\
        TQxNTdkYjZlMjJkYjAxYjIzNjdmYzhiNzMxZDA1MGJlNjBmNzQxMTZjNDIzNFwiLFwia2V5X2lkXCI6XCIxXCIsXCJzaWduXCI6XCIzMGJlNDJiN1wifS\
        IsInIiOiJodHRwczovL2JqLmxpYW5qaWEuY29tL3p1ZmFuZy9yY28zMS8iLCJvcyI6IndlYiIsInYiOiIwLjEifQ=='
    """
    USER_AGENT = ''
    COOKIE = ''

    def __init__(self, years, city, output_file):
        self.years = years
        self.city = city
        self.output_file = output_file

    def get_headers(self):
        return {
            'User-Agent': self.USER_AGENT,
            'Cookie': self.COOKIE
        }

    def generate_urls(self):
        """生成指定年份和城市的所有月份的天气页面链接"""
        base_url = 'http://lishi.tianqi.com/{}/{}{}.html'
        urls = []
        for year in self.years:
            urls.extend([base_url.format(self.city, year, f'{i:02}') for i in range(1, 13)])
        return urls

    def fetch_page(self, url):
        """获取页面内容"""
        headers = self.get_headers()
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            response.encoding = response.apparent_encoding
            return response.text
        return None

    def parse_weather_data(self, html):
        """解析天气数据"""
        soup = BeautifulSoup(html, 'html.parser')
        data = soup.find_all(class_='thrui')
        dates = re.findall(r'class="th200">(.*?)</', str(data))
        temps = re.findall(r'class="th140">(.*?)</', str(data))
        
        date_box = [date[:10] for date in dates]
        week_box = [date[10:] for date in dates]
        max_temp = [temps[i*4] for i in range(len(dates))]
        min_temp = [temps[i*4 + 1] for i in range(len(dates))]
        weather = [temps[i*4 + 2] for i in range(len(dates))]
        wind = [temps[i*4 + 3] for i in range(len(dates))]
        
        return pd.DataFrame({
            '日期': date_box,
            '星期': week_box,
            '最高温度': max_temp,
            '最低温度': min_temp,
            '天气': weather,
            '风向': wind
        })

    def save_to_csv(self, dataframe):
        """将数据保存到CSV文件"""
        dataframe.to_csv(self.output_file, encoding='utf_8_sig', index=False)

    def run(self):
        urls = self.generate_urls()
        
        all_data = []
        for url in urls:
            html = self.fetch_page(url)
            if html:
                df = self.parse_weather_data(html)
                all_data.append(df)
        
        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            self.save_to_csv(final_df)
            print(final_df)
        else:
            print("未能获取数据")

# 调用主函数
if __name__ == "__main__":
    scraper = WeatherScraper([2023, 2024], 'chongqing', '../data/weather_2023_2024_test.csv')
    scraper.run()