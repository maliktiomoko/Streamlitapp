
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
import streamlit as st
import pandas as pd
import io
import numpy as np
from src.tabular_forecasting.tabular_forecasting_utils import inverse_transform_predictions 
import matplotlib.pylab as plt
import subprocess
import time
import os

def generate_submission_and_metrics(sample_df, y_pred, y_train, y_tr, y_tr_pred, y_val, y_val_pred, preproc_candidates, target_cols, channel_id):
    """
    Generate submission DataFrame and compute validation metrics for multiple target columns.
    """
    
    if isinstance(target_cols, str):
        target_cols = [target_cols]
    
    # Inverse transform predictions if needed
    if preproc_candidates is None:
        pred_test_transformed = y_pred
    else:
        pred_test_transformed = []
        for i, preproc in enumerate(preproc_candidates):
            pred_col = y_pred[:, i] if y_pred.ndim > 1 else y_pred
            pred_test_transformed.append(inverse_transform_predictions(preproc, pred_col))
        pred_test_transformed = np.column_stack(pred_test_transformed)
    
    # Ensure shape is (n_samples, n_targets)
    if pred_test_transformed.ndim == 1:
        pred_test_transformed = pred_test_transformed[:, None]
    
    # Get submission ids
    if 'id' in sample_df.columns:
        submission_ids = sample_df['id']
    else:
        submission_ids = np.arange(pred_test_transformed.shape[0])
    submission_ids = np.array(submission_ids)
    
    # Adjust submission_ids if lengths mismatch
    if len(pred_test_transformed) != len(submission_ids):
        submission_ids = np.arange(len(pred_test_transformed))
    
    submission_dict = {"id": submission_ids}
    for i, col in enumerate(target_cols):
        submission_dict[col] = pred_test_transformed[:, i]
    submission = pd.DataFrame(submission_dict)
    
    buffer = io.StringIO()
    submission.to_csv(buffer, index=False)
    buffer.seek(0)
    
    # Mask rows with any NaN values in y_val
    mask = ~np.isnan(y_val).any(axis=1) if y_val.ndim > 1 else ~np.isnan(y_val)
    y_val_clean = y_val[mask]
    y_val_pred_clean = y_val_pred[mask]
    
    metrics = {}
    for i, col in enumerate(target_cols):
        y_val_i = y_val_clean[:, i] if y_val_clean.ndim > 1 else y_val_clean
        y_val_pred_i = y_val_pred_clean[:, i] if y_val_pred_clean.ndim > 1 else y_val_pred_clean
        
        metrics[col] = {
            "R²": r2_score(y_val_i, y_val_pred_i),
            "MSE": mean_squared_error(y_val_i, y_val_pred_i),
            "MAE": mean_absolute_error(y_val_i, y_val_pred_i),
            "RMSLE": np.sqrt(mean_squared_log_error(y_val_i, y_val_pred_i))
        }
    
    return submission, buffer, metrics, y_val_clean, y_val_pred_clean


def display_dashboard(submission, buffer, metrics, y_tr, y_tr_pred, y_val_clean, y_val_pred_clean):
    st.title("📈 Tabular Forecasting Dashboard")
    st.info("⚙️ Tabular data detected")

    tabs = st.tabs([
        "📊 Metrics",
        "📈 Training Curve",
        "🎯 Validation Predictions",
        "🔎 Residual Analysis",
        "🧪 Training Predictions",
        "📤 Submission"
    ])

    with tabs[0]:
        st.subheader("📊 Validation Metrics")
        st.table(metrics)

    with tabs[1]:
        st.subheader("🎯 Validation: True vs Predicted")
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_val_clean, y_val_pred_clean, alpha=0.4, color='green')
        ax2.plot([y_val_clean.min(), y_val_clean.max()], [y_val_clean.min(), y_val_clean.max()], 'r--', label='Ideal')
        ax2.set_xlabel('True')
        ax2.set_ylabel('Predicted')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)

    with tabs[2]:
        st.subheader("🔎 Residual Distribution (Validation Set)")
        residuals = y_val_clean - y_val_pred_clean
        fig3, ax3 = plt.subplots()
        ax3.hist(residuals, bins=30, color='coral', edgecolor='black')
        ax3.set_xlabel('Residual (True - Predicted)')
        ax3.set_ylabel('Frequency')
        ax3.grid(True)
        st.pyplot(fig3)

    with tabs[3]:
        st.subheader("🧪 Training Set: True vs Predicted")
        fig4, ax4 = plt.subplots()
        ax4.scatter(y_tr, y_tr_pred, alpha=0.4, color='purple')
        ax4.plot([y_tr.min(), y_tr.max()], [y_tr.min(), y_tr.max()], 'r--', label='Ideal')
        ax4.set_xlabel('True')
        ax4.set_ylabel('Predicted')
        ax4.legend()
        ax4.grid(True)
        st.pyplot(fig4)

    with tabs[4]:
        st.subheader("📤 Submission File Preview")
        st.dataframe(submission)
        st.download_button(
            label="📥 Download Prediction CSV",
            data=buffer.getvalue(),
            file_name="submission.csv",
            mime="text/csv"
        )


def submit_to_kaggle(submission, competition_name="store-sales-time-series-forecasting", config_dir=None):
    submission_file = "submission.csv"
    submission.to_csv(submission_file, index=False)

    if config_dir:
        os.environ["KAGGLE_CONFIG_DIR"] = config_dir

    st.info("📤 Submitting file to Kaggle...")
    result = subprocess.run([
        "kaggle", "competitions", "submit",
        "-c", competition_name,
        "-f", submission_file,
        "-m", "Auto submission from Streamlit"
    ], capture_output=True, text=True)

    st.code(result.stdout)

    if "Successfully submitted" in result.stdout:
        st.success("✅ Submitted successfully!")
        time.sleep(10)

        st.info("🔍 Checking latest submission and public score...")
        result = subprocess.run([
            "kaggle", "competitions", "submissions", "-c", competition_name
        ], capture_output=True, text=True)
        st.code(result.stdout)

        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            last_line = lines[-1]
            parts = last_line.split()
            if "pending" in last_line.lower():
                st.warning("⚠️ Submission is still being scored. Try again later.")
            elif len(parts) >= 5:
                score = parts[-2]
                st.success(f"🎯 Latest Public Score: `{score}`")
            else:
                st.warning("⚠️ Could not parse public score.")
        else:
            st.warning("⚠️ No submissions found.")
    else:
        st.error("❌ Submission failed. See logs above.")



# def submit_and_visualize(sample_df,y_pred, y_train, y_tr, y_tr_pred, y_val, y_val_pred, preproc_candidates, target_col):

#     submission_ids = sample_df['id'] if 'id' in sample_df.columns else np.arange(len(y_pred))
#     # target_col = y_train.name if hasattr(y_train, 'name') and y_train.name else "target"
#     # target_col = "sales"
#     if preproc_candidates is None:
#         pred_test_transformed = y_pred
#     else:
#         pred_test_transformed = inverse_transform_predictions(preproc_candidates[0], y_pred)
#     submission = pd.DataFrame({"id": submission_ids, target_col: pred_test_transformed})

#     buffer = io.StringIO()
#     submission.to_csv(buffer, index=False)
#     buffer.seek(0)

#     # Validation Metrics
#     mask = ~np.isnan(y_val)
#     y_val_clean = y_val[mask]
#     y_val_pred_clean = y_val_pred[mask]
#     st.title("📈 Tabular Forecasting Dashboard")
#     st.info("⚙️ Tabular data detected")
#     metrics = {
#         "R²": r2_score(y_val_clean, y_val_pred_clean),
#         "MSE": mean_squared_error(y_val_clean, y_val_pred_clean),
#         "MAE": mean_absolute_error(y_val_clean, y_val_pred_clean)
#     }

#     # 👉 Main Dashboard Tabs
#     tabs = st.tabs([
#         "📊 Metrics",
#         "📈 Training Curve",
#         "🎯 Validation Predictions",
#         "🔎 Residual Analysis",
#         "🧪 Training Predictions",
#         "📤 Submission"
#     ])

#     # Metrics tab
#     with tabs[0]:
#         st.subheader("📊 Validation Metrics")
#         st.table(metrics)


#     # Validation Predictions tab
#     with tabs[1]:
#         st.subheader("🎯 Validation: True vs Predicted")
#         fig2, ax2 = plt.subplots()
#         ax2.scatter(y_val_clean, y_val_pred_clean, alpha=0.4, color='green')
#         ax2.plot([y_val_clean.min(), y_val_clean.max()], [y_val_clean.min(), y_val_clean.max()], 'r--', label='Ideal')
#         ax2.set_xlabel('True num_sold')
#         ax2.set_ylabel('Predicted num_sold')
#         ax2.legend()
#         ax2.grid(True)
#         st.pyplot(fig2)

#     # Residual Analysis tab
#     with tabs[2]:
#         st.subheader("🔎 Residual Distribution (Validation Set)")
#         residuals = y_val_clean - y_val_pred_clean
#         fig3, ax3 = plt.subplots()
#         ax3.hist(residuals, bins=30, color='coral', edgecolor='black')
#         ax3.set_xlabel('Residual (True - Predicted)')
#         ax3.set_ylabel('Frequency')
#         ax3.set_title('Distribution of Residuals')
#         ax3.grid(True)
#         st.pyplot(fig3)

#     # Training Predictions tab
#     with tabs[3]:
#         st.subheader("🧪 Training Set: True vs Predicted")
#         fig4, ax4 = plt.subplots()
#         ax4.scatter(y_tr, y_tr_pred, alpha=0.4, color='purple')
#         ax4.plot([y_tr.min(), y_tr.max()], [y_tr.min(), y_tr.max()], 'r--', label='Ideal')
#         ax4.set_xlabel('True num_sold')
#         ax4.set_ylabel('Predicted num_sold')
#         ax4.legend()
#         ax4.grid(True)
#         st.pyplot(fig4)

#     # Submission tab
#     with tabs[4]:
#         st.subheader("📤 Submission File Preview")
#         st.dataframe(submission)
#         st.download_button(
#             label="📥 Download Prediction CSV",
#             data=buffer.getvalue(),
#             file_name="submission.csv",
#             mime="text/csv"
#         )

#         # Kaggle Auto-Submit UI
#         COMPETITION_NAME = "store-sales-time-series-forecasting"
#         KAGGLE_CONFIG_DIR = "/home/mtiomoko/post_training_forecasting_official-main/src/llm_data_preprocessing"
#         submission_file = "submission.csv"
#         submission.to_csv(submission_file, index=False)

        
#         os.environ["KAGGLE_CONFIG_DIR"] = KAGGLE_CONFIG_DIR
#         st.info("📤 Submitting file to Kaggle...")

#         result = subprocess.run([
#             "kaggle", "competitions", "submit",
#             "-c", COMPETITION_NAME,
#             "-f", submission_file,
#             "-m", "Auto submission from Streamlit"
#         ], capture_output=True, text=True)

#         st.code(result.stdout)

#         if "Successfully submitted" in result.stdout:
#             st.success("✅ Submitted successfully!")
#             time.sleep(10)

#             st.info("🔍 Checking latest submission and public score...")
#             result = subprocess.run([
#                 "kaggle", "competitions", "submissions", "-c", COMPETITION_NAME
#             ], capture_output=True, text=True)
#             st.code(result.stdout)

#             lines = result.stdout.strip().split("\n")
#             if len(lines) > 1:
#                 last_line = lines[-1]
#                 parts = last_line.split()
#                 if "pending" in last_line.lower():
#                     st.warning("⚠️ Submission is still being scored. Try again later.")
#                 elif len(parts) >= 5:
#                     score = parts[-2]
#                     st.success(f"🎯 Latest Public Score: `{score}`")
#                 else:
#                     st.warning("⚠️ Could not parse public score.")
#             else:
#                 st.warning("⚠️ No submissions found.")
#         else:
#             st.error("❌ Submission failed. See logs above.")