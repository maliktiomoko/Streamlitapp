import os
from dataclasses import dataclass


def setup_environment():
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["KAGGLE_CONFIG_DIR"] = "src/kaggle_utilities/"  # Replace with your actual path

@dataclass
class AppConfig:
    data_name: str
    model_name: str
    num_trials: int
    window_size: int
    prediction_horizon: int
    label_len: int
    n_samples: int
    feature_size: int
    model_file: str
    batch_size: int
    method: str
    n_jobs: int
    project_description: str
    stats: dict
