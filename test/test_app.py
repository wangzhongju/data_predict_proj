# -*- coding: utf-8 -*-
# @Date     : 2024/08/23
# @Author   : WZJ
# @File     : test_app.py

import os
import sys
import unittest
from pathlib import Path
from flask_testing import TestCase  # pip install Flask-Testing


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from app import create_app



# 设置测试用的 secret_key 和 salt_key
TEST_SECRET_KEY = 'test_secret_key'
TEST_SALT_KEY = 'test_salt_key'

class ForecastAppTestCase(TestCase):
    def create_app(self):
        app = create_app()
        app.config['SECRET_KEY'] = TEST_SECRET_KEY
        app.config['SALT_KEY'] = TEST_SALT_KEY
        return app

    def setUp(self):
        self.client = self.app.test_client()

    def test_forecast_all_sites_construction(self):
        # 用于测试的请求头
        headers = {
            'X-Client-Id': 'client1',
            'X-Token': self._generate_token()
        }
        
        # 用于测试的JSON数据
        data = {
            'periods': 4,
            'construction_sites': ['高新区西永组团F分区F03地块项目',
                                   '曾家“科研港”片区一路网工程'
                                   ],
            'disposal_sites': ['花都湖土地整治项目',
                               '玉龙二号消纳场'
                               ]
        }
        
        response = self.client.post('/algorithm/api/v1/data_prediction', json=data, headers=headers)
        self.assertEqual(response.status_code, 200)
        response_json = response.json
        self.assertEqual(response_json['code'], '0000')

    def test_get_token(self):
        response = self.client.get('/get_token')
        self.assertEqual(response.status_code, 200)
        response_json = response.json
        self.assertEqual(response_json['code'], '0000')
        self.assertIn('token', response_json['data'])

    def _generate_token(self):
        import hashlib
        return hashlib.md5((TEST_SECRET_KEY + TEST_SALT_KEY).encode()).hexdigest()

if __name__ == '__main__':
    unittest.main()