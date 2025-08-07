import streamlit as st
from src.app.ui import render_dataset_tab, render_forecasting_tab, render_feedback_tab
from src.app.config import setup_environment
from src.utilities.streamlit import setup_streamlit, setup_train_test_sample
from src.csv_merging.time_series_csv_merge import merge_process, generate_time_series_splits
from src.data_preprocessing.feature_construction import preprocess_process, train_test_process
from src.tabular_forecasting.tabular_forecasting_utils import train_tabular
from src.kaggle_utilities.kaggle_submit import generate_submission_and_metrics, display_dashboard, submit_to_kaggle
from src.time_series_training.training_time_series_utils import training_each_time_series
import os 
import pandas as pd
import numpy as np
import io
from src.post_training.reinforcement_learning import explore_instructions
from src.kaggle_utilities.prepare_submission import concatenate_predictions
from src.utilities.streamlit import sidebar_configuration
from src.post_training.testing import autoregressive_forecast

def main():
    # === Streamlit Page Setup ===
    tabs = setup_streamlit()

    # === Sidebar Configuration ===
    config = sidebar_configuration()

    # === Problem Description ===
    project_description = st.text_area("📝 Describe your forecasting or classification problem:", "")

    # === Session State Initialization ===
    for key in [
        "run_clicked", "confirmed", "merge_completed", "schemas", "dfs", "merged_df", "temp_ns",
        "merge_code", "sample_df", "sample_summary", "saved_train_paths", "saved_test_paths",
        "submission", "submit_now"
    ]:
        if key not in st.session_state:
            st.session_state[key] = False if key in ["confirmed", "merge_completed", "submit_now", "training_done"] else None

    # === Handle Submit-to-Kaggle Action ===
    if st.session_state.submit_now:
        st.session_state.submit_now = False  # Reset the flag
        if st.session_state.get("submission") is not None:
            with st.spinner("📤 Submitting to Kaggle..."):
                submit_to_kaggle(st.session_state.submission, config_dir="src/kaggle_utilities/", competition_name="store-sales-time-series-forecasting")
            st.success("✅ Submitted to Kaggle!")
        else:
            st.error("❌ No submission data found. Please run training first.")
        st.stop()
    # === Run Agent Button ===
    if st.button("🚀 Run Agent"):
        st.session_state.run_clicked = True
        st.session_state.confirmed = False
        st.session_state.merge_completed = False

    # === Step 1: Data Load & Merge ===
    if st.session_state.run_clicked and not st.session_state.merge_completed:
        if not config["train_files"]:
            st.error("❌ Please upload at least one training file to proceed.")
            st.stop()

        with st.spinner("⏳ Step 1: Analyzing training data..."):
            try:
                schemas, dfs, saved_train_paths, saved_test_paths, sample_df, sample_summary, test_split_info = setup_train_test_sample(
                    config["train_files"],
                    config["test_files"],
                    config["sample_submission_file"]
                )
                st.session_state.schemas = schemas
                st.session_state.dfs = dfs
                st.session_state.saved_train_paths = saved_train_paths
                st.session_state.saved_test_paths = saved_test_paths
                st.session_state.sample_df = sample_df
                st.session_state.sample_summary = sample_summary
            except Exception as e:
                st.error(f"❌ Failed to process uploaded data: {e}")
                st.stop()

        st.success("✅ Data successfully loaded.")
        st.info(f"ℹ️ Test Data: {test_split_info}")
        st.code(st.session_state.schemas, language="markdown")

        with st.spinner("🔗 Synthesizing merge code..."):
            pattern_list = []
            merged_df, temp_ns, merge_code, merge_error = merge_process(
                st.session_state.schemas,
                project_description,
                st.session_state.dfs,
                pattern_list
            )

        if merge_error:
            st.error(f"❌ Merge failed: {merge_error}") 
            st.stop()
        else:
            st.session_state.merged_df = merged_df
            st.session_state.temp_ns = temp_ns
            st.session_state.merge_code = merge_code
            st.session_state.merge_completed = True

    # === Step 2: Display Merged Data ===
    if st.session_state.merge_completed:
        with tabs[0]:
            st.subheader("📁 Dataset Merge")
            st.code(st.session_state.merge_code, language="python")
            st.success("✅ Merge successful!")
            st.dataframe(st.session_state.merged_df.head())

            # === Step 3: Group & Target Column Selection ===
            st.subheader("🧪 Configure Grouping and Target Column")
            df = st.session_state.merged_df


            grouping_cols = st.multiselect(
                "Select grouping column(s):",
                options=df.columns.tolist(),
                default=st.session_state.get("grouping_cols") or (["family"] if "family" in df.columns else []),
                key="grouping_cols"
            )

            target_cols = st.multiselect(
                "Select target column(s):",
                options=df.columns.tolist(),
                default=st.session_state.get("target_cols") or (["sales"] if "sales" in df.columns else []),
                key="target_cols"
            )
            st.subheader("📈 Select Exogenous Variable(s) (optional)")
            exogeneous_variable = st.multiselect(
                "Select exogenous variable(s):",
                options=df.columns.tolist(),
                default=[],
                key="exogeneous_variable"
            )

            if st.button("✅ Confirm Selection and Start Training"):
                if target_cols:
                    st.session_state.confirmed = True
                    if not grouping_cols:
                        st.info("No grouping columns selected. Proceeding without grouping.")
                else:
                    st.warning("Please select a target column.")

    # === Step 4: Training Workflow ===
    if st.session_state.confirmed and not st.session_state.get("training_done", False):

        st.session_state.training_done = True
        df = st.session_state.merged_df
        # print('dataframe', df)

        with st.spinner("📦 Generating time series splits..."):
            try:
                X_train_list, y_train_list, X_test_list, y_train_exo_list, y_test_exo_list, test_indices_list, group_keys, q_vec, channel_id = generate_time_series_splits(
                    final_df=df,
                    grouping_cols=st.session_state.grouping_cols,
                    target_cols=st.session_state.target_cols,
                    q=None,
                    exogeneous_variable=st.session_state.exogeneous_variable
                )
            except Exception as e:
                st.error(f"❌ Failed to generate splits: {e}")
                st.stop()

        st.success(f"✅ Successfully created splits for {len(group_keys)} groups.")

        with st.expander("📂 View a sample group split"):
            if X_train_list:
                st.markdown("### Sample Group")
                st.write(f"**Group key:** `{group_keys[0]}`")
                st.write(f"**Train shape:** {X_train_list[0].shape}")
                st.write(f"**Test shape:** {X_test_list[0].shape}")
                st.line_chart(X_train_list[0][0])
            else:
                st.warning("No valid training groups found.")

        pred_test = []
        max_sample = 100
        test_sample = sum([X_test_list[i].shape[1] for i in range(len(X_test_list))])
        with st.spinner("🏋️ Training models for each group..."):
            for i in range(len(X_train_list)):
                q = q_vec[i]
                X_train = X_train_list[i][:max_sample]
                y_train = y_train_list[i][:max_sample]
                X_test = X_test_list[i]
                # print('test shape', X_train.shape, y_train.shape, X_test.shape)
                # print(X_train_exo.shape, y_train_exo.shape, X_test_exo.shape)
                # print(X_train.shape, y_train.shape, X_test.shape)
                # print(X_train[:10, :10])
                # print(y_train[:10, :10])
                # print(X_test[:10, :10])

                st.markdown(f"### Training model for group `{group_keys[i]}`")

                try:
                    X_train_exo = y_train_exo_list[i] if y_train_exo_list[i] is not None else None
                    y_train_exo = y_train_exo_list[i] if y_train_exo_list[i] is not None else None
                    X_test_exo = y_test_exo_list[i]
                    print(X_train_exo.shape, y_train_exo.shape, X_test_exo.shape)
                    exp, args = training_each_time_series(
                        X_train, y_train, X_test,
                        X_train_exo=X_train_exo,
                        y_train_exo=y_train_exo,
                        X_test_exo=X_test_exo,
                        project_description=project_description,
                        saved_train_paths=st.session_state.saved_train_paths,
                        model_file=config["model_file"],
                        model_name=config["model_name"],
                        method=config["method"],
                        channel_id=channel_id
                    )
                    st.success(f"✅ Group `{group_keys[i]}` trained successfully.")
                except Exception as e:
                    st.error(f"❌ Training failed for group `{group_keys[i]}`: {e}")
                    continue

                post_training = False
                if post_training:
                    pred_rl, true, batch_x, best_functions = explore_instructions(exp, args, streamlit=False)
                    pred_final = autoregressive_forecast(exp, args, best_functions, [],
                                q, test_sample)
                else:
                    # args.model_id = 1
                    _, pred_rl, true, batch_x = exp.validation(args.model_id, test=1)
                    print('q', q)
                    pred_final = autoregressive_forecast(exp, args, [], [], q, test_sample)
                    best_functions = []

                pred_test.append(pred_final)

                # pred_test.append(pred_rl)
        # print('PRED', pred_test[0].shape)

        # === Step 5: Generate Submission & Display Metrics ===
        y_pred, y_tr_pred, y_tr, y_val_clean, y_val_pred_clean = concatenate_predictions(
        pred_test, test_indices_list, exp, args, target_cols=st.session_state.target_cols, channel_id=channel_id
)

        submission, buffer, metrics, y_val_clean, y_val_pred_clean = generate_submission_and_metrics(
            st.session_state.sample_df, y_pred, y_train, y_tr, y_tr_pred, y_val_clean, y_val_pred_clean,
            preproc_candidates=None, target_cols=st.session_state.target_cols, channel_id=channel_id
        )

        st.session_state.submission = submission

        display_dashboard(submission, buffer, metrics, y_tr, y_tr_pred, y_val_clean, y_val_pred_clean)
    
    if st.button("🚀 Submit to Kaggle"):
        st.session_state.submit_now = True
        st.session_state.training_done = True
        st.rerun()  # <-- force re-run so submission block runs immediately


if __name__ == "__main__":
    main()

