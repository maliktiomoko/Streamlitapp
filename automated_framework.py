import argparse
import os
import sys
import pandas as pd
import numpy as np
from src.csv_merging.time_series_csv_merge import merge_process, generate_time_series_splits
from src.time_series_training.training_time_series_utils import training_each_time_series
from src.kaggle_utilities.kaggle_submit import generate_submission_and_metrics
from src.kaggle_utilities.prepare_submission import concatenate_predictions
from src.post_training.reinforcement_learning import explore_instructions
from src.utilities.streamlit import setup_train_test_sample
from src.post_training.testing import autoregressive_forecast

import io

class FileLike:
    def __init__(self, path):
        self.path = path
        self.name = os.path.basename(path)
        with open(self.path, "rb") as f:
            self._bytes = f.read()

    def getbuffer(self):
        return memoryview(self._bytes)

def ask_user_choice(prompt, options):
    print(prompt)
    for i, opt in enumerate(options):
        print(f"{i+1}. {opt}")
    while True:
        choice = input(f"Enter choice number (1-{len(options)}): ")
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except:
            pass
        print("Invalid choice. Try again.")

def main():
    parser = argparse.ArgumentParser(description="Run Time Series Agent CLI")
    parser.add_argument("--inputs", nargs="+", help="Input CSV files or directory")
    parser.add_argument("--grouping-cols", nargs="*", help="Grouping column(s)")
    parser.add_argument("--target-cols", nargs="*", help="Target column(s)")
    parser.add_argument("--exogeneous_variable", nargs="*", help="Exogeneous column(s)")
    parser.add_argument("--output-dir", default="submissions", help="Output folder for submission CSV")
    parser.add_argument("--description", type=str, default="CLI run of Time Series Agent", help="Optional project description")

    args = parser.parse_args()

    # Detect if input is directory or files
    csv_files = []
    for path in args.inputs:
        if os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(path, f))
        elif os.path.isfile(path) and path.endswith(".csv"):
            csv_files.append(path)
        else:
            print(f"Warning: {path} is not a CSV file or directory, ignoring.")

    if not csv_files:
        print("Error: No CSV files detected. Exiting.")
        sys.exit(1)

    # Simple detection: Look for train/test/sample files by keywords
    test_files = [f for f in csv_files if "test" in f.lower()]
    sample_sub_files = [f for f in csv_files if "sample" in f.lower() or "submission" in f.lower()]

    # Train files are everything else
    train_files = [
        f for f in csv_files
        if f not in test_files and f not in sample_sub_files
    ]

    # Fallback if no train/test distinction
    if not train_files:
        train_files = csv_files
    if not sample_sub_files:
        sample_sub_files = [None]

    print(f"Detected train files: {train_files}")
    print(f"Detected test files: {test_files}")
    print(f"Detected sample submission files: {sample_sub_files}")

    # Load train, test and sample submission DataFrames
    train_dfs = [pd.read_csv(f) for f in train_files]
    test_dfs = [pd.read_csv(f) for f in test_files] if test_files else []
    sample_df = pd.read_csv(sample_sub_files[0]) if sample_sub_files[0] else None

    # Setup train/test/sample paths (mimic your existing code)
    saved_train_paths = train_files
    saved_test_paths = test_files
    sample_submission_file = sample_sub_files[0]

    # Compose schemas and dfs as your code expects
    # (Assuming your setup_train_test_sample handles this — else you can mimic)
    schemas = [list(df.columns) for df in train_dfs]
    # dfs = train_dfs + test_dfs
    def wrap_paths_as_filelike(paths):
        return [FileLike(p) for p in paths]

    train_files_wrapped = wrap_paths_as_filelike(train_files)
    test_files_wrapped = wrap_paths_as_filelike(test_files) if test_files else []
    sample_sub_files_wrapped = wrap_paths_as_filelike(sample_sub_files) if sample_sub_files[0] else [None]

    schemas, dfs, saved_train_paths, saved_test_paths, sample_df, sample_summary, test_split_info = setup_train_test_sample(
        train_files_wrapped,
        test_files_wrapped,
        sample_sub_files_wrapped[0]
    )


    # Project description (could be a CLI arg or just blank)
    project_description = args.description

    # Merge Process
    print("Merging data...")
    pattern_list = []
    merged_df, temp_ns, merge_code, merge_error = merge_process(schemas, project_description, dfs, pattern_list)
    if merge_error:
        print(f"Merge failed: {merge_error}")
        sys.exit(1)

    print("Data merged successfully.")

    # Ask user for grouping columns if not provided
    if args.grouping_cols:
        grouping_cols = args.grouping_cols
    else:
        print("\nAvailable columns for grouping:")
        grouping_cols = []
        if len(merged_df.columns) > 0:
            choice = ask_user_choice("Select grouping column (or enter 0 for none):", ["None"] + merged_df.columns.tolist())
            if choice != "None":
                grouping_cols = [choice]

    # Ask user for target columns if not provided
    if args.target_cols:
        target_cols = args.target_cols
    else:
        print("\nAvailable columns for target:")
        target_cols = [ask_user_choice("Select target column:", merged_df.columns.tolist())]
    print(target_cols)
    
    # Ask user for exogeneous variables (multiple allowed)
    if args.exogeneous_variable:
        exogeneous_variable = args.exogeneous_variable  # assumed to already be a list of indices
    else:
        print("\nAvailable columns for exogeneous variables (select multiple by number, comma-separated, or 0 for none):")
        for i, col in enumerate(merged_df.columns):
            print(f"{i + 1}. {col}")
        
        exo_input = input("Enter column numbers (e.g., 1,3,5) or 0 for none: ").strip()
        
        if exo_input == "0":
            exogeneous_variable = []
        else:
            selected_indices = []
            for item in exo_input.split(","):
                item = item.strip()
                if item.isdigit():
                    idx = int(item) - 1  # convert to 0-based index
                    if 0 <= idx < len(merged_df.columns):
                        selected_indices.append(idx)
                    else:
                        print(f"Ignoring out-of-range index: {item}")
                else:
                    print(f"Ignoring invalid input: {item}")
            
            exogeneous_variable = selected_indices
    print("Generating time series splits...")
    try:
        X_train_list, y_train_list, X_test_list, y_train_exo_list, y_test_exo_list, test_indices_list, group_keys, q_vec, channel_id = generate_time_series_splits(
            final_df=merged_df,
            grouping_cols=grouping_cols,
            target_cols=target_cols if len(target_cols) == 1 else target_cols,
            q=None,
            exogeneous_variable=exogeneous_variable
        )
    except Exception as e:
        print(f"Error during time series split generation: {e}")
        sys.exit(1)

    print(f"Created splits for {len(group_keys)} groups.")
    # Training models for each group
    pred_test = []
    test_sample = sum([y_train_list[i].shape[1] for i in range(len(y_train_list))])
    # print(test_sample)
    for i in range(len(X_test_list)):
        q = q_vec[i]
        print(f"Training model for group {group_keys[i]}...")
        try:
            print(X_train_list[i].shape, y_train_list[i].shape, X_test_list[i].shape)
            print(y_train_exo_list[i].shape, y_test_exo_list[i].shape)
            exp, args_ = training_each_time_series(
                X_train_list[i],
                y_train_list[i],
                X_test_list[i],
                X_train_exo=y_train_exo_list[i] if y_train_exo_list[i] is not None else None,
                y_train_exo=y_train_exo_list[i] if y_train_exo_list[i] is not None else None,
                X_test_exo=y_test_exo_list[i],
                project_description=project_description,
                saved_train_paths=saved_train_paths,
                model_file=None,
                model_name='DLinear',
                method="random", channel_id=channel_id
            )
        except Exception as e:
            print(f"Training failed for group {group_keys[i]}: {e}")
            continue

        # Post training RL or normal validation
        post_training = False
        if post_training:
            pred_rl, true, batch_x, best_functions = explore_instructions(exp, args_, streamlit=False)
            args.model_id = 1
            pred_final = autoregressive_forecast(exp, args, best_functions, [],
                        q, test_sample)
        else:
            args.model_id = 1
            # print(q)
            _, pred_rl, true, batch_x = exp.validation(args_.model_id, test=1)
            pred_final = autoregressive_forecast(exp, args, [], [], q, test_sample)
            best_functions = []

        # print(pred_final.shape)

        pred_test.append(pred_final[0, :, channel_id])

    # Concatenate predictions
    y_pred, y_tr_pred, y_tr, y_val_clean, y_val_pred_clean = concatenate_predictions(
        pred_test, test_indices_list, exp, args_, target_cols=target_cols, channel_id=channel_id
    )

    # Generate submission and metrics
    submission, buffer, metrics, y_val_clean, y_val_pred_clean = generate_submission_and_metrics(
        sample_df, y_pred, y_train_list, y_tr, y_tr_pred, y_val_clean, y_val_pred_clean,
        preproc_candidates=None, target_cols=target_cols, channel_id=channel_id
    )

    # Save submission file
    os.makedirs(args.output_dir, exist_ok=True)
    submission_path = os.path.join(args.output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

    print("Metrics:")
    for col, metric_dict in metrics.items():
        print(f"Column: {col}")
        for metric_name, value in metric_dict.items():
            print(f"  {metric_name}: {value:.4f}")

        # Ask if the user wants to submit to Kaggle
    submit_choice = ask_user_choice(
        "\nDo you want to submit the generated submission to Kaggle?",
        ["Yes", "No"]
    )

    if submit_choice == "Yes":
        try:
            from src.kaggle_utilities.kaggle_submit import submit_to_kaggle
            submit_to_kaggle(
                submission,
                config_dir="src/kaggle_utilities/",
                competition_name="store-sales-time-series-forecasting"
            )
            print("✅ Submission uploaded to Kaggle.")
        except Exception as e:
            print(f"❌ Submission to Kaggle failed: {e}")
    else:
        print("Skipping Kaggle submission.")

if __name__ == "__main__":
    main()
