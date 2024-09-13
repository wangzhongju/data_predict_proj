

import sys
import yaml
import json
import hashlib
from pathlib import Path
from flask import Flask, request, jsonify
from src.construction_sites_manager import ConstructionSitesManager

def add_root_to_sys_path():
    """Add the root directory to sys.path."""
    file_path = Path(__file__).resolve()
    root_dir = file_path.parents[0]
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))



class ForecastApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.config_path = Path(__file__).resolve().parents[0] / "config/config.yaml"
        self.config = self.load_config(self.config_path)
        self.manager = self.initialize_manager(self.config)
        self.secret_key = 'test_secret_key'
        self.salt_key = 'test_salt_key'
        self.valid_client_ids = {'client1', 'client2'}

        self.setup_routes()

    def setup_routes(self):
        self.app.add_url_rule('/algorithm/api/v1/data_prediction',
                              'forecast_all_sites_construction',
                              self.forecast_all_sites_construction, methods=['GET', 'POST']
                        )
        self.app.add_url_rule('/get_token', 'get_token', self.get_token, methods=['GET'])

    def load_config(self, path):
        """Load configuration from a YAML file."""
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def initialize_manager(self, config):
        """Initialize the ConstructionSitesManager based on the configuration."""
        construction_file = config.get('construction_file')
        disposal_file = config.get('disposal_file')
        manager = ConstructionSitesManager(construction_file, disposal_file, config)
        # Initialize sites based on request
        manager.init_sites()
        manager.add_data_to_sites_construction()
        manager.calculate_all_remaining_capacities_construction()
        manager.add_data_to_sites_disposal()
        manager.calculate_all_remaining_capacities_disposal()
        return manager

    def calculate_md5(self, key, salt):
        return hashlib.md5((key + salt).encode()).hexdigest()

    def verify_token(self, token):
        expected_token = self.calculate_md5(self.secret_key, self.salt_key)
        print("expected_token: ", expected_token)
        return token == expected_token

    def verify_client(self, client_id):
        return client_id in self.valid_client_ids

    def generate_temporary_token(self):
        return 'e2311e2f32e4fe81d6f6dfad29b5ab44'

    def forecast_all_sites_construction(self):
        client_id = request.headers.get('X-Client-Id')
        token = request.headers.get('X-Token')

        if not self.verify_client(client_id):
            return jsonify({'code': '4002', 'message': 'Invalid client ID', 'data': {}}), 400

        if not self.verify_token(token):
            return jsonify({'code': '4001', 'message': 'Invalid or missing token', 'data': {}}), 401

        try:
            data = request.json
            periods = data.get('periods', 4)
            construction_sites = data.get('construction_sites', [])
            disposal_sites = data.get('disposal_sites', [])

            self.manager.forecast_all_sites_construction_app(periods=periods, construction_name=construction_sites)
            self.manager.forecast_all_sites_disposal_app(periods=periods, disposal_name=disposal_sites)

            results = {}
            for site_name, site in self.manager.sites_construction.items():
                if site_name in construction_sites:
                    df = site.data_resampled_future.copy()
                    for col in df.select_dtypes(include=['datetime']):
                        df[col] = df[col].dt.strftime('%Y-%m-%d')
                    df_json = df.to_json(orient='split', date_format='iso')
                    results[site_name] = json.loads(df_json)
            for site_name, site in self.manager.sites_disposal.items():
                if site_name in disposal_sites:
                    df = site.data_resampled_future.copy()
                    for col in df.select_dtypes(include=['datetime']):
                        df[col] = df[col].dt.strftime('%Y-%m-%d')
                    df_json = df.to_json(orient='split', date_format='iso')
                    results[site_name] = json.loads(df_json)

            return jsonify({'code': '0000', 'message': 'Success', 'data': results})
        except Exception as e:
            return jsonify({'code': '5000', 'message': f'Internal server error: {str(e)}', 'data': {}}), 500

    def get_token(self):
        token = self.generate_temporary_token()
        return jsonify({'code': '0000', 'message': 'Token generated successfully', 'data': {'token': token}})


def create_app():
    app_instance = ForecastApp()
    return app_instance.app




if __name__ == '__main__':
    add_root_to_sys_path()
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False)