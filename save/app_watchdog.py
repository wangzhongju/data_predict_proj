

import sys
import yaml
import json
# import threading
from pathlib import Path
from flask import Flask, request, jsonify

# config
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from src.construction_sites_manager import ConstructionSitesManager

def add_root_to_sys_path():
    """Add the root directory to sys.path."""
    file_path = Path(__file__).resolve()
    root_dir = file_path.parents[0]
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_manager(config):
    """Initialize ConstructionSitesManager with configuration."""
    construction_file = config.get('construction_file')
    disposal_file = config.get('disposal_file')
    forecast_periods = config.get('forecast_periods', 8)  # Default forecast periods
    weather_file = config.get('weather_file')
    output_file = config.get('output_file')
    
    return ConstructionSitesManager(construction_file, disposal_file, config)

class ConfigFileHandler(FileSystemEventHandler):
    def __init__(self, config_path, manager_ref):
        self.config_path = config_path
        self.manager_ref = manager_ref

    def on_modified(self, event):
        if event.src_path == str(self.config_path):
            print("Config file changed. Reloading...")
            new_config = load_config(self.config_path)
            self.manager_ref['manager'] = initialize_manager(new_config)

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Load configuration
    config_path = Path(__file__).resolve().parents[0] / "config/config.yaml"
    config = load_config(config_path)
    
    # Initialize manager
    global manager
    manager = initialize_manager(config)

    # Watchdog setup
    config_file_handler = ConfigFileHandler(config_path, {'manager': manager})
    observer = Observer()
    observer.schedule(config_file_handler, path=str(config_path.parent), recursive=False)
    observer.start()

    @app.route('/forecast', methods=['GET', 'POST'])
    def forecast_all_sites_construction():
        """Forecast construction capacities for all sites."""
        data = request.json
        periods = data.get('periods', 4)
        print("====debug: get 'periods' = ", periods)

        # Perform forecasting
        manager.add_data_to_sites_construction()
        manager.calculate_all_remaining_capacities_construction()
        manager.forecast_all_sites_construction(periods=periods)
        manager.plot_all_sites_gd()

        manager.plot_save()
        
        # # Assuming self.data_resampled_future contains the results you want to return
        # results = {site_name: site.data_resampled_future for site_name, site in manager.sites_construction.items()}

        # Convert DataFrames to JSON serializable format
        results = {}
        for site_name, site in manager.sites_construction.items():
            # Convert DataFrame to JSON
            df_json = site.data_resampled_future.to_json(orient='split')
            print("df_json: ", df_json)
            results[site_name] = json.loads(df_json)  # Parse JSON string into Python dict

        return jsonify(results)
    
    @app.teardown_appcontext
    def shutdown_observer(exception=None):
        observer.stop()
        observer.join()

    return app

if __name__ == '__main__':
    add_root_to_sys_path()
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)





# from flask import Flask

# app = Flask(__name__)

# @app.route('/')
# def home():
#     return 'Hello, World!'

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000, debug=True)