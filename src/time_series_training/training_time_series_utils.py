
from sklearn.model_selection import train_test_split
import numpy as np
import streamlit as st
from src.utilities.utils import ensure_3d
from src.csv_merging.csv_description_and_utilities import describe_csvs
from torch.utils.data import DataLoader
from src.data_preprocessing.feature_construction import drop_non_numeric_features
from src.csv_merging.csv_description_and_utilities import TimeSeriesForecastingDataset
from src.time_series_models.model_extraction import define_model

def time_series_split(X, y, test_size=0.3):
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    X_val = X[split_index:]
    y_train = y[:split_index]
    y_val = y[split_index:]
    return X_train, X_val, y_train, y_val


def training_each_time_series(
    X_train, y_train, X_test,
    X_train_exo=None, y_train_exo=None, X_test_exo=None,
    project_description=None, saved_train_paths=None,
    model_file=None, model_name=None, method=None, channel_id=None
):
    # print('HERE 1')
    stats = describe_csvs(saved_train_paths)

    # Split train/val for both main and exogeneous data
    X_train_main, X_val_main, y_train_main, y_val_main = time_series_split(X_train, y_train, test_size=0.5)

    if X_train_exo is not None and y_train_exo is not None:
        X_train_exo, X_val_exo, y_train_exo, y_val_exo = time_series_split(X_train_exo, y_train_exo, test_size=0.5)
    else:
        # Create dummy exogeneous data if not provided
        X_train_exo = X_val_exo = y_train_exo = y_val_exo = None

    X_test_main = X_test
    X_test_exo = X_test_exo

    # Ensure 3D shape for all arrays
    to_3d = lambda arr: ensure_3d(arr) if arr is not None else None
    X_train_main, y_train_main, X_val_main, y_val_main, X_test_main = map(to_3d, [X_train_main, y_train_main, X_val_main, y_val_main, X_test_main])
    if X_train_exo is not None:
        X_train_exo, y_train_exo, X_val_exo, y_val_exo, X_test_exo = map(to_3d, [X_train_exo, y_train_exo, X_val_exo, y_val_exo, X_test_exo])

    y_test_main = np.zeros((X_test_main.shape[0], y_val_main.shape[1], y_val_main.shape[2]))
    if y_train_exo is not None:
        y_test_exo = np.zeros((X_test_exo.shape[0], y_val_exo.shape[1], y_val_exo.shape[2]))
    else:
        y_test_exo = None

    # print('HERE 2')

    # Broadcast y arrays if only one feature to match X feature dim for main data
    def broadcast_y(arr, target_dim):
        if arr.shape[2] == 1:
            return np.broadcast_to(arr, (arr.shape[0], arr.shape[1], target_dim)).copy()
        else:
            return arr

    # y_train_main = broadcast_y(y_train_main, X_train_main.shape[2])
    # y_val_main = broadcast_y(y_val_main, X_val_main.shape[2])
    # y_test_main = broadcast_y(y_test_main, X_test_main.shape[2])

    # if y_train_exo is not None:
    #     y_train_exo = broadcast_y(y_train_exo, X_train_exo.shape[2])
    #     y_val_exo = broadcast_y(y_val_exo, X_val_exo.shape[2])
    #     y_test_exo = broadcast_y(y_test_exo, X_test_exo.shape[2])

    dummy_marks = lambda x: np.zeros((x.shape[0], x.shape[1], 4))

    # Drop non-numeric features for all data
    X_train_main = drop_non_numeric_features(X_train_main)
    X_val_main = drop_non_numeric_features(X_val_main)
    y_train_main = drop_non_numeric_features(y_train_main)
    y_val_main = drop_non_numeric_features(y_val_main)
    X_test_main = drop_non_numeric_features(X_test_main)
    y_test_main = drop_non_numeric_features(y_test_main)

    if X_train_exo is not None:
        X_train_exo = drop_non_numeric_features(X_train_exo)
        X_val_exo = drop_non_numeric_features(X_val_exo)
        y_train_exo = drop_non_numeric_features(y_train_exo)
        y_val_exo = drop_non_numeric_features(y_val_exo)
        if X_test_exo is not None:
            X_test_exo = drop_non_numeric_features(X_test_exo)
            y_test_exo = drop_non_numeric_features(y_test_exo)

    # print('HERE 3')y_t
    # Create data loaders with exogeneous data
    # print('y_train_exo', y_train_exo.shape, y_val_exo.shape, X_test_exo.shape)
    train_dataset = TimeSeriesForecastingDataset(
        X_train_main, y_train_main,
        X_exo=X_train_exo, y_exo=y_train_exo
    )
    val_dataset = TimeSeriesForecastingDataset(
        X_val_main, y_val_main,
        X_exo=X_val_exo, y_exo=y_val_exo
    )
    test_dataset = TimeSeriesForecastingDataset(
        X_test_main, y_test_main,
        X_exo=X_test_exo, y_exo=X_test_exo
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # print('HERE 4')
    config = {
        "data_name": saved_train_paths[0],
        "models": model_name,
        "num_trials": 1,
        "window_size": X_train_main.shape[1],        # seq_len
        "prediction_horizon": y_train_main.shape[1], # pred_len
        "label_len": 0,
        "n_samples": 2000,
        "feature_size": X_train_main.shape[-1],       # enc_in
        "exog_dim": y_train_exo.shape[2] if X_train_exo is not None else 0,  # exog_dim for exogenous
        "task_name": "long_term_forecast",             # adjust as needed
        "dropout": 0.1,
        "num_class": 10,
        "model_file": model_file,
        "batch_size": 32,
        "method": method,
        "n_jobs": 7,
        "project_description": project_description,
        "stats": stats,
        "channel_id": channel_id,
    }
    # print('HERE 5')

    exp, args = define_model(train_loader, val_loader, test_loader, config)
    # print('HERE 6')
    args.project_description = project_description
    args.stats = stats
    args.channel_id = channel_id
    # print('args.', args.exog_dim)
    exp.train(args)

    return exp, args

