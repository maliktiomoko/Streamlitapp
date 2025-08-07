import streamlit as st
import requests
from streamlit_lottie import st_lottie
import pandas as pd
from src.utilities.utils import save_uploaded_files
from src.tabular_forecasting.tabular_forecasting_utils import summarize_df, summarize_csvs
import os



def initialize_session_state():
    keys = [
        'feedback', 'phase', 'best_pred_rl', 'true_feedback', 'batch_x_feedback',
        'best_function_set_per_channel', 'best_function_set_per_channel_feedback',
        'new_class', 'exp', 'args', 'is_tabular'
    ]
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = None
    if 'phase' not in st.session_state or st.session_state.phase is None:
        st.session_state.phase = 1


def sidebar_configuration():
    st.sidebar.header("🛠️ Configuration")

    train_files = st.sidebar.file_uploader("📁 Upload Training CSV Files", type=["csv"], accept_multiple_files=True)
    test_files = st.sidebar.file_uploader("🧪 Upload Test CSV Files (Optional)", type=["csv"], accept_multiple_files=True)
    sample_submission_file = st.sidebar.file_uploader("📝 Upload Sample Submission (Optional)", type=["csv"])

    model_name = st.sidebar.selectbox("🤖 Model Name", [
        "Crossformer", "DLinear", "ETSformer", "FEDformer", "FiLM", "FreTS", "Informer", "Koopa",
        "LightTS", "MICN", "Nonstationnary_Transformer", "PatchTST", "Pyraformer", "Reformer", "SegRNN",
        "TSMixer", "TiDE", "TimeMixer", "Transformer", "TimesNet", "iTransformer"
    ])

    model_file = st.sidebar.file_uploader("🧩 Upload Custom Model (Optional)", type=["py"])
    window_size = st.sidebar.number_input("🪟 Sliding Window Size", min_value=1, max_value=100000, value=96)
    prediction_horizon = st.sidebar.number_input("🔮 Prediction Horizon", min_value=1, max_value=100000, value=144)
    method = st.sidebar.selectbox("🧪 Optimization Method", ["random", "SH-HPO", "Genetic"])

    return {
        "train_files": train_files,
        "test_files": test_files,
        "sample_submission_file": sample_submission_file,
        "model_name": model_name,
        "model_file": model_file,
        "window_size": window_size,
        "prediction_horizon": prediction_horizon,
        "method": method
    }


def load_lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def setup_streamlit():
    initialize_session_state()
    st.title("🔁 Time Series AI Agent")
    st.markdown("#### The future of forecasting: no-code, self-optimizing, feedback-driven AI.")
    st.caption("Built by GTSBrain Huawei | Powered by OpenAI & InternVL | June 2025")

    lottie_ai = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_tno6cg2w.json")
    if lottie_ai:
        st_lottie(lottie_ai, height=150, key="header_animation")

    with st.expander("📽 How to Use This App"):
        st.markdown("Upload your dataset, describe your problem, and let the AI agent do the rest!")

    tabs = st.tabs(["📁 Dataset", "📈 Forecasting", "🧠 Agent Feedback"])
    return tabs


def setup_train_test_sample(train_files, test_files, sample_submission_file):
    if not train_files:
        raise ValueError("No training files provided.")

    saved_train_paths = save_uploaded_files(train_files)

    # Handle test files or auto-split
    if not test_files:
        full_train_df = pd.concat([pd.read_csv(p) for p in saved_train_paths], axis=0)
        split_index = int(len(full_train_df) * 0.8)
        df_train, df_test = full_train_df.iloc[:split_index], full_train_df.iloc[split_index:]

        train_temp_path, test_temp_path = "temp_data/auto_train.csv", "temp_data/auto_test.csv"
        df_train.to_csv(train_temp_path, index=False)
        df_test.to_csv(test_temp_path, index=False)
        saved_train_paths, saved_test_paths = [train_temp_path], [test_temp_path]
        test_split_info = "Auto-split applied (80/20)."
    else:
        saved_test_paths = save_uploaded_files(test_files)
        test_split_info = "Test files used."

    # Handle sample submission
    if sample_submission_file:
        saved_sample_path = save_uploaded_files([sample_submission_file])[0]
        sample_df = pd.read_csv(saved_sample_path)
        sample_summary = summarize_df(sample_df, "sample_submission.csv")
    else:
        sample_df = pd.DataFrame()
        sample_summary = "No submission file."

    dfs = {
        os.path.splitext(os.path.basename(f))[0]: pd.read_csv(
            f,
            parse_dates=['date'] if 'date' in pd.read_csv(f, nrows=1).columns else None
        )
        for f in saved_train_paths + saved_test_paths
    }

    csv_paths = saved_train_paths + saved_test_paths
    schemas = summarize_csvs(csv_paths)

    return schemas, dfs, saved_train_paths, saved_test_paths, sample_df, sample_summary, test_split_info
            