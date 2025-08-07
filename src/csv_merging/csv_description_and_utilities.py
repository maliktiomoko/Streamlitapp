import pandas as pd
import os
import numpy as np
import streamlit as st


import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from typing import List
import numpy as np

import os
from typing import List
import pandas as pd
import numpy as np

def extract_schema_from_csvs(csv_paths: List[str], max_rows_preview: int = 3) -> str:
    """
    Extracts a resilient and LLM-ready markdown schema from multiple CSV files, optimized for time series modeling and transformation generation.

    Args:
        csv_paths (List[str]): List of CSV file paths for training or testing.
        max_rows_preview (int): Number of preview rows to include per file.

    Returns:
        str: Structured markdown schema of the datasets.
    """
    schema_md = "# Time Series Dataset Schema Overview\n\n"

    for path in csv_paths:
        filename = os.path.basename(path)
        schema_md += f"## `{filename}`\n\n"

        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            schema_md += f"⚠️ `{filename}` is empty and was skipped.\n\n---\n\n"
            continue
        except Exception as e:
            schema_md += f"❌ Failed to read `{filename}`: {type(e).__name__}: {e}\n\n---\n\n"
            continue

        schema_md += f"**Shape**: {df.shape[0]} rows × {df.shape[1]} columns\n\n"
        try:
            file_size_kb = os.path.getsize(path) / 1024
            schema_md += f"**File Size**: {file_size_kb:.2f} KB\n\n"
        except:
            schema_md += "**File Size**: Unknown\n\n"

        # Column summary
        schema_md += "### Column Summary\n"
        schema_md += "| Column | Type | Nulls | Null % | Unique | Sample Values |\n|--------|------|-------|--------|--------|----------------|\n"
        for col in df.columns:
            try:
                dtype = df[col].dtype
                nulls = df[col].isnull().sum()
                null_pct = (nulls / len(df)) * 100
                unique = df[col].nunique()
                sample_vals = df[col].dropna().astype(str).unique()[:3]
                sample_str = ", ".join([v[:30] + '…' if len(v) > 30 else v for v in sample_vals])
                schema_md += f"| {col} | {dtype} | {nulls} | {null_pct:.2f}% | {unique} | {sample_str} |\n"
            except Exception as e:
                schema_md += f"| {col} | ❌ Error | - | - | - | {e} |\n"

        # Descriptive statistics
        schema_md += "\n### Descriptive Statistics (Numeric Columns)\n\n"
        try:
            numeric_desc = df.describe(include=[np.number])
            schema_md += numeric_desc.to_markdown() + "\n"
        except:
            schema_md += "_Could not compute descriptive statistics._\n"

        # Data type distribution
        schema_md += "\n### Data Types Overview\n\n"
        try:
            dtype_counts = df.dtypes.value_counts()
            schema_md += "| Data Type | Count |\n|-----------|-------|\n"
            for dtype, count in dtype_counts.items():
                schema_md += f"| {dtype} | {count} |\n"
        except:
            schema_md += "_Could not summarize data types._\n"

        # Missing values
        schema_md += "\n### Missing Value Summary\n\n"
        try:
            missing = df.isnull().sum()
            if missing.sum() > 0:
                schema_md += "| Column | Missing |\n|--------|---------|\n"
                for col, val in missing.items():
                    if val > 0:
                        schema_md += f"| {col} | {val} |\n"
            else:
                schema_md += "No missing values detected.\n"
        except:
            schema_md += "_Could not analyze missing values._\n"

        # Timestamp detection
        schema_md += "\n### Temporal Column Inference\n\n"
        try:
            datetime_candidates = []
            for col in df.columns:
                try:
                    if pd.to_datetime(df[col], errors='raise', utc=True).notnull().mean() > 0.8:
                        datetime_candidates.append(col)
                except:
                    continue

            if datetime_candidates:
                ts_col = datetime_candidates[0]
                ts_series = pd.to_datetime(df[ts_col])
                schema_md += f"- Candidate timestamp column: `{ts_col}`\n"
                schema_md += f"- Range: {ts_series.min()} → {ts_series.max()}\n"
                schema_md += f"- Sorted: {'✅' if ts_series.is_monotonic_increasing else '❌'}\n"
            else:
                schema_md += "- ⚠️ No reliable timestamp column found.\n"
        except:
            schema_md += "- ❌ Timestamp inference failed.\n"

        # Data preview
        schema_md += f"\n### Preview of First {max_rows_preview} Rows\n\n"
        try:
            schema_md += df.head(max_rows_preview).to_markdown(index=False) + "\n"
        except:
            schema_md += "_Data preview unavailable._\n"

        schema_md += "\n---\n\n"

    return schema_md

def detect_datetime_column(df):
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        if df[col].dtype == object:
            try:
                pd.to_datetime(df[col])
                return col
            except Exception:
                continue
    return None
class TimeSeriesForecastingDataset(Dataset):
    def __init__(self, X, y, X_exo=None, y_exo=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # self.seq_x_mark = torch.tensor(seq_x_mark, dtype=torch.float32)
        # self.seq_y_mark = torch.tensor(seq_y_mark, dtype=torch.float32)
        
        if X_exo is not None:
            self.X_exo = torch.tensor(X_exo, dtype=torch.float32)
        else:
            self.X_exo = None
            
        if y_exo is not None:
            self.y_exo = torch.tensor(y_exo, dtype=torch.float32)
        else:
            self.y_exo = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.X_exo is not None and self.y_exo is not None:
            return (
                self.X[idx],
                self.y[idx],
                # self.seq_x_mark[idx],
                # self.seq_y_mark[idx],
                self.X_exo[idx],
                self.y_exo[idx]
            )
        else:
            return (
                self.X[idx],
                self.y[idx],
                # self.seq_x_mark[idx],
                # self.seq_y_mark[idx]
            )

    
def describe_csvs(file_paths):
    import pandas as pd

    if not file_paths:
        return {"columns": {}, "error": "No files provided"}

    stats = {"columns": {}}

    try:
        sample_df = pd.read_csv(file_paths[0])
    except Exception as e:
        return {"columns": {}, "error": f"Failed to read CSV: {str(e)}"}

    if sample_df.empty:
        return {"columns": {}, "error": "CSV is empty"}

    for col in sample_df.columns:
        dtype = str(sample_df[col].dtype)
        stats["columns"][col] = dtype

    return stats

def parse_natural_description(description, stats):
    """
    Parse user description and stats to extract a robust forecasting config.
    Fallbacks included to ensure functionality even with vague or empty prompts.
    """
    columns = stats.get("columns", {})
    if not columns:
        raise ValueError("No columns found in stats. Please check CSV parsing.")

    numerical_columns = [col for col, dtype in columns.items() if "float" in dtype or "int" in dtype]
    datetime_columns = [col for col, dtype in columns.items() if "datetime" in dtype or "date" in dtype]

    # Fallback if description is missing or vague
    if not description.strip():
        st.warning("⚠️ No natural language description provided. Using default configuration.")
        target_column = numerical_columns[-1] if numerical_columns else "target"
        feature_columns = numerical_columns[:-1] if len(numerical_columns) > 1 else []
        context_length = 96
        horizon = 144
    else:
        # Use LLM or basic NLP parsing (or placeholder for now)
        # This is where a real NLP step would go — now we simulate it.
        target_column = None
        for col in numerical_columns:
            if "target" in col.lower() or "value" in col.lower() or "y" == col.lower():
                target_column = col
                break

        if not target_column and numerical_columns:
            target_column = numerical_columns[-1]

        feature_columns = [col for col in numerical_columns if col != target_column]
        context_length = 96
        horizon = 144

    # Fallback if LLM didn’t specify any useful name
    if target_column is None:
        target_column = "target_" + str(len(numerical_columns))
        st.warning(f"⚠️ Could not infer target column — assigning default name: `{target_column}`")

    if not feature_columns:
        feature_columns = [f"feat_{i}" for i in range(len(numerical_columns) - 1)]

    config = {
        "target_column": target_column,
        "feature_columns": feature_columns,
        "context_length": context_length,
        "horizon": horizon,
        "datetime_column": datetime_columns[0] if datetime_columns else None
    }
    if "date_column" not in config or config["date_column"] not in stats.get("columns", {}):
        # Try to detect one from stats
        for col, dtype in stats.get("columns", {}).items():
            if "date" in col.lower() or "time" in col.lower():
                config["date_column"] = col
                break

    return config

def prepare_forecasting_data(train_paths, test_paths, parsed_config, window_size, prediction_horizon):
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from src.automated_preprocessing import detect_datetime_column  # Adjust as needed

    # Load data
    train_df = pd.concat([pd.read_csv(p) for p in train_paths], axis=0).reset_index(drop=True)
    test_df = pd.concat([pd.read_csv(p) for p in test_paths], axis=0).reset_index(drop=True) if test_paths else None

    # Use sample_df for inspection
    sample_df = train_df.copy()

    # Detect or create datetime column
    date_col = parsed_config.get("date_column")
    if date_col is None or date_col not in sample_df.columns:
        date_col = detect_datetime_column(sample_df)

    if date_col is None:
        print("[Info] No datetime column found. Creating synthetic datetime...")
        synthetic_start = pd.Timestamp("2000-01-01")
        sample_df["synthetic_date"] = pd.date_range(start=synthetic_start, periods=len(sample_df), freq="H")
        train_df["synthetic_date"] = pd.date_range(start=synthetic_start, periods=len(train_df), freq="H")
        if test_df is not None:
            test_df["synthetic_date"] = pd.date_range(
                start=synthetic_start + pd.Timedelta(hours=len(train_df)),
                periods=len(test_df),
                freq="H"
            )
        date_col = "synthetic_date"

    # Ensure datetime format
    train_df[date_col] = pd.to_datetime(train_df[date_col])
    train_df = train_df.sort_values(by=date_col).reset_index(drop=True)
    if test_df is not None:
        test_df[date_col] = pd.to_datetime(test_df[date_col])
        test_df = test_df.sort_values(by=date_col).reset_index(drop=True)

    # Detect features/target
    target = parsed_config.get("target_column")
    features = parsed_config.get("feature_columns")

    numeric_cols = sample_df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != date_col]

    if not features:
        features = [col for col in numeric_cols if col != target]

    if not target:
        target = features[-1] if features else numeric_cols[-1]
        features = [col for col in numeric_cols if col != target]

    numerical_cols = features + [target]

    # Scale numerical columns
    scaler = StandardScaler()
    train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
    if test_df is not None:
        test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])

    def create_windows(df, context_length, horizon):
        n_samples = min(len(df) - context_length - horizon, 100)  # Optional: limit for speed

        # Pre-extract numeric and datetime
        data = df[numerical_cols].values
        dt = df[date_col].dt

        # Time features
        months, days, weekdays, hours = dt.month.values, dt.day.values, dt.weekday.values, dt.hour.values

        # Allocate
        X = np.zeros((n_samples, context_length, len(numerical_cols)))
        y = np.zeros((n_samples, horizon, len(numerical_cols)))
        seq_x_mark = np.zeros((n_samples, context_length, 4))
        seq_y_mark = np.zeros((n_samples, horizon, 4))

        for i in range(n_samples):
            s, ec, et = i, i + context_length, i + context_length + horizon
            X[i] = data[s:ec]
            y[i] = data[ec:et]

            seq_x_mark[i, :, 0] = months[s:ec]
            seq_x_mark[i, :, 1] = days[s:ec]
            seq_x_mark[i, :, 2] = weekdays[s:ec]
            seq_x_mark[i, :, 3] = hours[s:ec]

            seq_y_mark[i, :, 0] = months[ec:et]
            seq_y_mark[i, :, 1] = days[ec:et]
            seq_y_mark[i, :, 2] = weekdays[ec:et]
            seq_y_mark[i, :, 3] = hours[ec:et]

        return X, y, seq_x_mark, seq_y_mark

    print("[Info] Creating training windows...")
    X_train, y_train, seq_x_train, seq_y_train = create_windows(train_df, window_size, prediction_horizon)

    if test_df is not None:
        print("[Info] Creating test windows...")
        X_test, y_test, seq_x_test, seq_y_test = create_windows(test_df, window_size, prediction_horizon)
    else:
        X_train, X_test, y_train, y_test, seq_x_train, seq_x_test, seq_y_train, seq_y_test = train_test_split(
            X_train, y_train, seq_x_train, seq_y_train, test_size=0.2, random_state=42
        )

    return X_train, y_train, seq_x_train, seq_y_train, X_test, y_test, seq_x_test, seq_y_test, train_df, test_df



