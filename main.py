

import argparse
import json
import yaml
from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from src.construction_sites_manager import ConstructionSitesManager
from data.data_process import DataProcessor, DataProcessorXNC




def main(config_path):
    # 读取配置文件
    # with open(config_path, 'r') as file:
    #     config = json.load(file)

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # 从配置文件中提取参数
    construction_file = config.get('construction_file')
    disposal_file = config.get('disposal_file')
    forecast_periods = config.get('forecast_periods', 8)  # 默认预测8周
    weather_file = config.get('weather_file')
    output_file = config.get('output_file')

    # # data process
    # processor = DataProcessor(weather_file, construction_file)
    # processor.run(output_file)

    # processor_ = DataProcessorXNC(weather_file, disposal_file)
    # processor_.run(output_file)

    # 初始化管理器
    manager = ConstructionSitesManager(construction_file, disposal_file, config)

    manager.init_sites()
    
    # # 工地处理
    # manager.add_data_to_sites_construction()
    # manager.calculate_all_remaining_capacities_construction()
    # manager.forecast_all_sites_construction(periods=forecast_periods)  # 预测
    # manager.plot_all_sites_gd()

    # # 消纳场处理
    # manager.add_data_to_sites_disposal()
    # manager.calculate_all_remaining_capacities_disposal()
    # manager.forecast_all_sites_disposal(periods=forecast_periods)  # 预测
    # manager.plot_all_sites_xnc()


    
    # 工地处理
    manager.add_data_to_sites_construction()
    manager.calculate_all_remaining_capacities_construction()
    sigal = manager.forecast_all_sites_construction_app(periods=forecast_periods,
                                                   construction_name=["高新区西永组团F分区F03地块项目",
                                                                      "高新区西永组团Z分区Z46地块项目",
                                                                      "寨山坪生态居住小区三期"
                                                                     ])  # 预测

    # # 消纳场处理
    # manager.add_data_to_sites_disposal()
    # manager.calculate_all_remaining_capacities_disposal()
    # # manager.forecast_all_sites_disposal(periods=forecast_periods)  # 预测
    # sigal = manager.forecast_all_sites_disposal_app(periods=forecast_periods,
    #                                                disposal_name=["花都湖土地整治项目",
    #                                                               "玉龙二号消纳场",
    #                                                              ])  # 预测

    # 显示
    if sigal:
        manager.plot_all_sites_gd_app()
        # manager.plot_all_sites_xnc_app()
        manager.plot_show()




if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='Process data using a configuration file.')
    parser.add_argument('--config', type=str, default=ROOT / "config/config.yaml", help='Path to the configuration file')
    
    args = parser.parse_args()
    
    # 调用主函数
    main(args.config)
