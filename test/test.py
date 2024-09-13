

import requests



# Base URL of the Flask app
BASE_URL = 'http://192.168.10.54:5000'

def get_token():
    response = requests.get(f'{BASE_URL}/get_token')
    if response.status_code == 200:
        return response.json()['data']['token']
    else:
        print('Failed to get token:', response.text)
        return None

def forecast(token):
    headers = {
        'X-Client-Id': 'client1',
        'X-Token': token
    }
    data = {
        'periods': 4,
        'construction_sites': ['高新区西永组团F分区F03地块项目',
                               '曾家“科研港”片区一路网工程'
                              ],
        'disposal_sites': ['花都湖土地整治项目',
                           '玉龙二号消纳场'
                          ]
    }
    response = requests.post(f'{BASE_URL}/algorithm/api/v1/data_prediction', json=data, headers=headers)
    if response.status_code == 200:
        print('Response:', response.json())
    else:
        print('Failed to data_prediction:', response.text)

if __name__ == '__main__':
    token = get_token()  # test_token: e2311e2f32e4fe81d6f6dfad29b5ab44
    if token:
        forecast(token)
