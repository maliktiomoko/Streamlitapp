import os
import numpy as np
import pandas as pd
import streamlit as st
import lightgbm as lgb
from sklearn.preprocessing import OneHotEncoder
from .utils import compute_metrics, plot_true_vs_pred
from src.csv_merging.time_series_csv_merge import summarize_csvs, synthesize_merge_code, split_candidates, exec_code_with_imports, clean_code
from src.tabular_forecasting.tabular_forecasting_utils import (
    summarize_final_df, synthesize_preprocessing_code,
    synthesize_train_test_split_code, clean_code_preprocess,
    summarize_df
)

def train_lightgbm_model(X_train, y_train, X_val, y_val, params=None):
    model = lgb.LGBMRegressor(**(params or {}))
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='l1',
        verbose=False
    )
    return model

def run_tabular_forecasting(description):
    data_dir = "/home/mtiomoko/post_training_forecasting_official-main/data/challenge1/"
    csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    csv_paths = [os.path.join(data_dir, f) for f in csv_files]
    dfs = {
        f.replace('.csv',''): pd.read_csv(
            os.path.join(data_dir, f),
            parse_dates=['date'] if 'date' in pd.read_csv(os.path.join(data_dir, f), nrows=1).columns else None
        )
        for f in csv_files
    }
    schemas = summarize_csvs(csv_paths)
    st.code(schemas, language="markdown")

    sample_path = os.path.join(data_dir, "sample_submission.csv")
    sample_df = pd.read_csv(sample_path) if os.path.exists(sample_path) else pd.DataFrame()
    sample_summary = summarize_df(sample_df, "sample_submission.csv") if not sample_df.empty else "No submission file."

    with st.spinner("🔗 Synthesizing merge code..."):
        raw_merge_code = synthesize_merge_code(schemas, description, "Tabular forecasting pipeline", n_candidates=1)
        merge_candidates = split_candidates(raw_merge_code)
        st.code(merge_candidates[0], language="python")

        temp_ns = dict(dfs)
        temp_ns.update({'pd': pd, 'np': np, 'OneHotEncoder': OneHotEncoder})

        try:
            exec_code_with_imports(clean_code(merge_candidates[0]), temp_ns)
            merged_df = temp_ns['final_df']
            st.dataframe(merged_df.head())
        except Exception as e:
            st.error(f"Merge failed: {e}")
            st.stop()

    with st.spinner("🧹 Synthesizing preprocessing code..."):
        df_summary = summarize_final_df(merged_df)
        raw_preproc = synthesize_preprocessing_code(df_summary, "Tabular forecasting pipeline", [], n_candidates=1)
        preproc_candidates = split_candidates(raw_preproc)
        st.code(preproc_candidates[0], language="python")

        try:
            temp_ns2 = dict(temp_ns)
            exec_code_with_imports(clean_code_preprocess(preproc_candidates[0]), temp_ns2)
            processed_df = temp_ns2['processed_df']
            st.dataframe(processed_df.head())
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            st.stop()

    with st.spinner("✂️ Synthesizing train/test split..."):
        df_summary = summarize_final_df(processed_df)
        raw_split_code = synthesize_train_test_split_code(
            df_summary=df_summary,
            description= description,
            sample_submission_summary=sample_summary,
            n_candidates=1
        )
        split_candidates_list = split_candidates(raw_split_code)
        st.code(split_candidates_list[0], language="python")

        try:
            temp_ns3 = dict(temp_ns2)
            exec_code_with_imports(clean_code(split_candidates_list[0]), temp_ns3)
            X_train, y_train, X_test = temp_ns3['split_train_test'](processed_df)
            st.dataframe(X_train.head())
        except Exception as e:
            st.error(f"Split failed: {e}")
            st.stop()

    columns_to_drop = X_test.isna().mean()[lambda x: x > 0.5].index
    if len(columns_to_drop) >= len(X_test.columns):
        st.error("Too many columns dropped. Dataset became empty.")
        st.stop()

    X_train = X_train.drop(columns=columns_to_drop)
    X_test = X_test.drop(columns=columns_to_drop)

    common_cols = sorted(set(X_train.columns) & set(X_test.columns) - {'id', 'date'})
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]

    for df in [X_train, X_test]:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

    split_idx = int(len(X_train) * 0.8)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

    hyper_params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': ['l1'],
        'learning_rate': 0.1,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.7,
        'bagging_freq': 10,
        'verbose': -1,
        'max_depth': 50,
        'num_leaves': 128,
        'max_bin': 512
    }

    with st.spinner("🏋️ Training LightGBM Model..."):
        model = train_lightgbm_model(X_tr, y_tr, X_val, y_val, hyper_params)

    y_val_pred = np.clip(model.predict(X_val), 0, None)
    metrics = compute_metrics(y_val, y_val_pred)
    st.write(metrics)
    plot_true_vs_pred(y_val, y_val_pred, title="Validation: True vs Predicted")
