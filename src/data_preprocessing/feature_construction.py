import numpy as np
import streamlit as st
from src.tabular_forecasting.tabular_forecasting_utils import (
    summarize_final_df, synthesize_preprocessing_code, split_candidates,
    exec_code_with_imports, clean_code_preprocess, synthesize_train_test_split_code, clean_code
)
def drop_non_numeric_features(X: np.ndarray) -> np.ndarray:
    """
    Remove all features in the last axis of a NumPy array X that are not numeric.
    Works for 2D or 3D arrays (e.g., (N, T, D) or (N, D)).
    """
    X = np.asarray(X)
    if X.ndim not in [2, 3]:
        raise ValueError(f"Input must be 2D or 3D array, but got shape {X.shape}")

    D = X.shape[-1]
    valid_mask = []

    for i in range(D):
        feature = X[..., i]
        # Try converting to float
        try:
            feature.astype(np.float32)
            if not np.issubdtype(feature.dtype, np.number):
                raise ValueError
            valid_mask.append(True)
        except:
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)

    if not np.any(valid_mask):
        raise ValueError("No valid numeric features remain after filtering.")

    X = X[..., valid_mask]
    return X.astype(np.float32)


def preprocess_process(merged_df, project_description, preproc_patterns, temp_ns):
    with st.spinner("🧹 Synthesizing preprocessing code..."):
        df_summary = summarize_final_df(merged_df)
        raw_preproc = synthesize_preprocessing_code(df_summary, project_description, preproc_patterns, n_candidates=1)
        preproc_candidates = split_candidates(raw_preproc)
        st.code(preproc_candidates[0], language="python")

        try:
            temp_ns2 = dict(temp_ns)
            exec_code_with_imports(clean_code_preprocess(preproc_candidates[0]), temp_ns2)
            processed_df = temp_ns2['processed_df']
            st.write("✅ Preview of processed dataframe:")
            st.dataframe(processed_df.head())
        except Exception as e:
            st.error(f"❌ Preprocessing failed: {e}")
            st.stop()
    return processed_df, temp_ns2, preproc_candidates

def train_test_process(processed_df, temp_ns2, project_description, sample_summary):
    with st.spinner("✂️ Synthesizing train/test split..."):
        df_summary = summarize_final_df(processed_df)
        raw_split_code = synthesize_train_test_split_code(
            df_summary=df_summary,
            description=project_description,
            sample_submission_summary=sample_summary,
            n_candidates=1
        )
        split_candidates_list = split_candidates(raw_split_code)
        st.code(split_candidates_list[0], language="python")

        try:
            temp_ns3 = dict(temp_ns2)
            exec_code_with_imports(clean_code(split_candidates_list[0]), temp_ns3)
            X_train, y_train, X_test = temp_ns3['split_train_test'](processed_df)

            st.success("✅ Split complete!")
            st.write(f"📊 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}")
            st.dataframe(X_train.head())
        except Exception as e:
            st.error(f"❌ Split failed: {e}")
            st.stop()
    return X_train, y_train, X_test, temp_ns3