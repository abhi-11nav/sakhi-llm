import os

import yaml


class SakhiConfig:
    def __init__(self, d):
        for k, v in d.items():
            if isinstance(v, dict):
                v = SakhiConfig(v)
            self.__dict__[k] = v

    @staticmethod
    def _load_config(config_path: str):
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        config = SakhiConfig(config_dict)

        config.paths.log_dir = os.path.join(config.paths.save_dir, "logs_dir")
        config.paths.model_dir = os.path.join(config.paths.save_dir, "model_dir")

        return config
