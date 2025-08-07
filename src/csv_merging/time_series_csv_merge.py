import os, re, traceback
import pandas as pd, numpy as np
import lightgbm as lgb
import torch, torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI
import warnings
import ast
import streamlit as st
from sklearn.preprocessing import OneHotEncoder
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.impute import SimpleImputer

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import warnings

PREDEFINED_CODE = """import pandas as pd

### Candidate 1 ###
train_df = train.copy()
test_df = test.copy()

# Merge train and test to keep 'id' column
train_df['is_test'] = 0
test_df['is_test'] = 1

# Combine train and test data
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Merge with oil data
combined_df = pd.merge(combined_df, oil, on='date', how='left')

# Merge with transactions data
combined_df = pd.merge(combined_df, transactions, on=['date', 'store_nbr'], how='left')

# Merge with stores data
combined_df = pd.merge(combined_df, stores, on='store_nbr', how='left')

# Merge with holidays_events data
combined_df = pd.merge(combined_df, holidays_events, on='date', how='left')

final_df = combined_df
"""


def clean_inputs(X_train, y_train, X_test):
    # Convert to DataFrame if needed
    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    # if isinstance(y_train, np.ndarray):
    #     y_train = pd.Series(y_train.flatten())

    # Remove non-numeric columns (e.g., datetime, objects)
    X_train = X_train.select_dtypes(include=[np.number])
    X_test = X_test.select_dtypes(include=[np.number])

    # Drop rows with NaNs in training data
    # valid_rows = ~y_train.isna() & X_train.notna().all(axis=1)
    # X_train = X_train[valid_rows]
    # y_train = y_train[valid_rows]

    # Replace remaining NaNs in test with 0 (or another strategy)
    X_test = X_test.fillna(0)

    return X_train.values, y_train, X_test.values

def train_and_run(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    X_train_clean, y_train_clean, X_test_clean = clean_inputs(X_train, y_train, X_test)
    
    model = LinearRegression()
    # print('test', X_train_clean.shape, y_train_clean.shape, X_test_clean.shape)
    model.fit(X_train_clean, y_train_clean)
    y_pred = model.predict(X_test_clean)
    # print('ypred', y_pred.shape)
    return y_pred

def generate_time_series_subseries(df, description, patterns, n_candidates=1):
    # This function calls your LLM prompt generator, e.g.,
    # generate_fully_automated_time_series_split_code, with the dataset sample and description
    raw_code = generate_fully_automated_time_series_split_code(
        df.head(100),  # or a sample of your data
        description,
        patterns,
        n_candidates=n_candidates
    )
    # Extract code
    code_str = split_candidates(raw_code)[0]
    ns = {"final_df": df, "pd": pd, "np": np, "T": 100}  # provide context
    exec_code_with_imports(clean_code(code_str), ns)
    # Expected variables after execution:
    # - X_train_list
    # - y_train_list
    # - X_test_list
    return ns["X_train_list"], ns["y_train_list"], ns["X_test_list"]


MODEL_NAME = "qwen2.5-coder-32b-instruct"


def exec_code_with_imports(code_str, namespace):
    """
    Extract import lines and exec them first, then exec the rest.
    """
    import_lines = []
    code_lines = []

    for line in code_str.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            import_lines.append(line)
        else:
            code_lines.append(line)

    # Execute imports first
    exec("\n".join(import_lines), namespace)
    # Then execute the rest of the code
    exec("\n".join(code_lines), namespace)


def call_openai(content, model=MODEL_NAME, temperature=0.3, max_tokens=10000):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()

import re

def clean_code_preprocess(code_str):
    # Remove markdown code fences
    code_str = re.sub(r'```python|```', '', code_str)
    
    cleaned_lines = []
    for line in code_str.split('\n'):
        stripped = line.strip()
        
        if stripped == '':
            cleaned_lines.append(line)
            continue
        
        if stripped.startswith('#'):
            cleaned_lines.append(line)
            continue
        
        # Allow indented lines
        if line.startswith(' ') or line.startswith('\t'):
            cleaned_lines.append(line)
            continue
        
        # Allow lines starting with common keywords
        if re.match(r'^(import|from|def|class|for|while|if|elif|else|try|except|with|return|break|continue|pass|raise|print)\b', stripped):
            cleaned_lines.append(line)
            continue
        
        # Allow lines that contain '=' or '(' which cover assignments and function calls
        if '=' in line or '(' in line:
            cleaned_lines.append(line)
            continue
        
        # Otherwise skip lines (likely plain English)
        # print(f"Skipping line: {line}")  # Uncomment for debug
    
    return '\n'.join(cleaned_lines)


def clean_code(code_str: str) -> str:
    # Strip only the outermost triple backticks (the wrapping)
    lines = code_str.strip().splitlines()
    
    # Remove the first and last lines *only if* they are code fences
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines)


def synthesize_train_test_split_code(df_summary, description, sample_submission_summary=None, n_candidates=3):
    prompt = f"""
        You are a Kaggle Grandmaster-level data scientist.

        You are given a pandas DataFrame called `processed_df`. It contains both training and test rows.
        Here is a summary of the DataFrame:
        {df_summary}

        Competition description (if relevant):
        {description}

        {"You are also given a sample submission format:" if sample_submission_summary else ""}
        {sample_submission_summary if sample_submission_summary else ""}

        Your task is to write {n_candidates} different Python code snippets that:
        - Split `processed_df` into:
            - `X_train`: features for training
            - `y_train`: the target column (inferred from sample submission or column names)
            - `X_test`: features for test set (rows to be submitted)
        - Ensure the `X_test` is aligned with the submission format (correct order, number of rows)
        - Avoid data leakage (no future information in training)
        - Use datetime columns, unique IDs, or columns like is_test (0 or 1), or patterns from competitions to determine the train/test split
        - If `id` is present, preserve it for reordering `X_test`; do not drop it unless you reattach it later

        ⚠️ IMPORTANT:
        - Return valid Python code only
        - Wrap the split logic inside a function:
            ```python
            def split_train_test(processed_df):
                ...
                return X_train, y_train, X_test
            ```
        - Do NOT include explanations or comments
        - Each candidate should be separated by:
            ### Candidate n ###
        """
    return call_openai(prompt)

def split_candidates(code_str):
    # Step 1: Extract all import lines from the top
    lines = code_str.splitlines()
    import_lines = []
    start_index = 0

    for i, line in enumerate(lines):
        if line.strip().startswith("import ") or line.strip().startswith("from "):
            import_lines.append(line)
        elif line.strip().startswith("### Candidate"):
            start_index = i
            break

    # Step 2: Rebuild the rest of the code (excluding top imports)
    code_after_imports = "\n".join(lines[start_index:])

    # Step 3: Split by Candidate block
    parts = re.split(r"### Candidate\s*\d*\s*###", code_after_imports)
    candidates = []

    for part in parts:
        part = part.strip()
        if not part:
            continue

        full_code = "\n".join(import_lines) + "\n\n" + part
        candidates.append(full_code.strip())

    return candidates

def train_and_predict(X_train, y_train, X_test, X_val=None, y_val=None):
    exclude_cols = {'date', 'id'}
    common_cols = list(set(X_train.columns).intersection(X_test.columns) - exclude_cols)
    common_cols.sort()

    # Subset and sanitize features
    X_train_sel = X_train[common_cols].copy()
    X_test_sel = X_test[common_cols].copy()
    X_train_sel, name_map = sanitize_column_names(X_train_sel)
    X_test_sel = X_test_sel.rename(columns=name_map)

    if X_val is None or y_val is None:
        X_tr, X_val, y_tr, y_val = train_test_split(X_train_sel, y_train, test_size=0.2, random_state=42)
    else:
        X_tr, y_tr = X_train_sel, y_train
        X_val = X_val[common_cols].rename(columns=name_map)

    # LightGBM Dataset
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.03,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'verbosity': -1,
        'n_jobs': -1,
        'seed': 42
    }

    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=500,
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(0)  # 0 disables logging, or set to 50 for periodic logging
        ]
    )

    y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
    val_mse = mean_squared_error(y_val, y_pred_val)

    pred_test = model.predict(X_test_sel, num_iteration=model.best_iteration)

    return pred_test, val_mse

# --- Config & Client ---
MODEL_NAME = "qwen2.5-coder-32b-instruct"
OPENAI_API_BASE = "http://api.openai.ukrc.huawei.com:4000/v1"
OPENAI_API_KEY = "sk-1234"
client = OpenAI(base_url=OPENAI_API_BASE, api_key=OPENAI_API_KEY)

def call_openai(prompt, temperature=0.3, max_tokens=10000):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"user","content":prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()

# --- Kaggle setup ---
def setup_kaggle():
    import os
    config_path = "/home/mtiomoko/post_training_forecasting_official-main/src/llm_data_preprocessing"
    kaggle_json = os.path.join(config_path, "kaggle.json")
    
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(f"Expected kaggle.json at: {kaggle_json}")
    
    os.environ["KAGGLE_CONFIG_DIR"] = config_path
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api

# --- LSTM modules ---
class TimeSeriesLSTM(nn.Module):
    def __init__(self, in_size, hidden=64, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(in_size, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)
    def forward(self, x):
        return self.out(self.lstm(x)[0][:, -1])

def train_lstm(X, y, epochs=10, lr=0.001):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).view(-1,1)
    model = TimeSeriesLSTM(X.shape[2])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(model(X_t), y_t)
        loss.backward(); opt.step()
    return model

def synthesize_time_series_code(df_summary, description, patterns, n_candidates=1):
    prompt = f"""
    You are a world-class time series forecasting expert and Python code assistant.

    You are working with a pandas DataFrame called `final_df`, already loaded in memory.

    ---

    ## 📄 1. Data Summary:
    {df_summary}

    ## 🎯 2. Project Description:
    {description}

    ## 🧠 3. Reference Code Patterns:
    These are examples of high-performing solutions from Kaggle:
    {patterns}

    ---

    ## 🧩 Your Goal:
    Write {n_candidates} complete and executable Python script(s) that do **end-to-end time series forecasting** on `final_df`, using best practices.

    ---

    ### 🧠 Pipeline Requirements:

    1. **Column detection**:
    - Auto-detect the time column (e.g. `date`)
    - Identify target column (e.g. `sales`, `y`)
    - Detect group columns (e.g. `store_nbr`, `family`) if they exist

    2. **Feature identification**:
    - Static covariates (e.g. store type)
    - Past covariates (e.g. transactions)
    - Future covariates (e.g. holidays, oil prices, etc.)
    - Apply good judgment for separation based on names/types

    3. **Preprocessing**:
    - Fill missing values
    - Apply log1p transform (with `InvertibleMapper`)
    - Use `Scaler()` for normalization
    - One-hot encode static covariates if categorical

    4. **Time series formatting**:
    ✅ Use `TimeSeries.from_group_dataframe()` with only supported arguments:
    - `df`, `group_cols`, `time_col`, `value_cols`, `freq`, `fill_missing_dates`, `fillna_value`.
    # Remove duplicate timestamps within groups by aggregating
    if final_df.duplicated(subset=group_cols + [time_col]).any():
        final_df = (
            final_df.groupby(group_cols + [time_col], as_index=False)
            .mean(numeric_only=True)
        )
    🚫 Do NOT use `static_covariates_cols` — it is not a valid keyword argument and will cause an error.

    ✅ To assign static covariates, loop over each `TimeSeries` after creation and call `.with_static_covariates(df)` manually.

    - Preserve group/static covariates

    5. **Feature engineering**:
    - Create lags (t-1, t-7, etc.)
    - Rolling averages (e.g. 7/14/28-day means)
    - Calendar features (day of week, month, etc.)
    - Holiday / payday flags (if available)

    6. **Model training**:
    - Use a Darts global model like `NHiTSModel`, `BlockRNNModel`, or `TransformerModel`
    - Apply cross-validation if needed
    - Handle “zero rule”: if last `N` target values are zero, forecast zero

    7. **Output**:
    - Final transformed time series: `processed_series`
    - Trained model: `trained_model`
    - Prediction (optional): `forecast`

    ---

    ### 🔒 Hard Constraints:
    ✅ DO NOT perform file loading/saving — assume final_df is already in memory

    ✅ Include all necessary import statements

    ✅ Do not output markdown, text, or explanations — only valid Python code

    ✅ Catch possible issues like missing columns with if "column_name" in final_df.columns checks

    🔁 Output Format:
    Provide {n_candidates} complete Python code block(s) separated like this:

    Candidate 1
    <code>
    Candidate 2
    <code>
    Only include clean and runnable code. No markdown or commentary.

    Now write the code.
    """
    response_ = call_openai(prompt)
    # print('RESPONSE:', response_)
    return response_

# --- Utility & preprocessing functions (merge + preprocess agents, sanitizing, etc.) ---
def sanitize_column_names(df):
    return df.rename(columns={c: re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns}), None

def synthesize_merge_code(schemas, patterns, description, n_candidates=1):
    content = f"""
        You are a Pandas expert. You have these pandas DataFrames already loaded, with variable names matching the filenames without extensions (that you should infer):

        {schemas}

        Based on optional actual Kaggle merge patterns:

        {patterns}

        And the project goal:

        {description}

        Write {n_candidates} different clean, leakage-safe Pandas code snippets to merge these DataFrames appropriately for the forecasting task.

        IMPORTANT:
        - Do NOT load CSV files again.
        - If any DataFrame (like test or sample_submission) includes an `id` column, KEEP it in `final_df`.
        - If there is no `id`, try to keep a column or combination of columns that uniquely identifies each row
        - Create a column named is_test that contains whether or not this row is part of the test sample(1 or 0) to help for futher train/test split
        - Use the variables as given.
        - Assign the final merged DataFrame to a variable named `final_df`.
        - INCLUDE all necessary import statements (like pandas).
        - Return only executable Python code, without comments or explanations.
        - Separate each candidate by a line with exactly: ### Candidate n ###
        - Check for existence of columns to avoid errors like ['type'] not in index". The code should work without any error
        """
    response_ = call_openai(content)
    # print('RESPONSE FROM LLM', response_)
    return response_


def generate_fully_automated_time_series_split_code(df_sample, description, patterns, n_candidates=1):
    content = f"""
You are an expert machine learning engineer. You must write **robust** and **fully automated** Python code to process a pandas DataFrame called `final_df`.

This DataFrame contains time series data for multiple entities (e.g., grouped by store or product category), and your task is to prepare it for supervised learning.

---

## Your Code Must:

1. **Infer the structure of `final_df`**
   - Print or infer all column names and types.
   - Automatically detect:
     - The timestamp column (dtype `datetime64[ns]`).
     - Categorical columns (dtype `object` or `category`).
     - Numeric columns.
    A suitable grouping key:

    Must be a categorical column (i.e., dtype object or category) that:

    Has moderate cardinality: fewer unique values than 20% of total rows.

    Has at least one value appearing in > 100 rows.

    Bonus points if the column name includes keywords like 'store', 'product', 'region', 'series', or 'dept'.

    Compute a score per candidate based on:

    Low-to-moderate cardinality.

    High frequency of values.

    Optional name heuristics.

    Choose the candidate with the highest score.

    If no column passes all criteria, fall back to:

    The only categorical column (if there is just one).

    Or, if no grouping column is found, treat the dataset as a single time series by adding:
    final_df['_dummy_group'] = 0
    grouping_col = '_dummy_group'


2. **Do not hard-code column names** like `'sales'`, `'is_test'`, `'type_x'`, etc.
   - **Only use columns that are actually present**, by checking:
     ```python
     if 'sales' in final_df.columns:
         # then proceed
     ```
Automatically infer the forecast horizon `q`:
   - If the `'is_test'` column is present:
     - Infer `q` as the **minimum number of test rows** across all groups.
       Example:
       ```python
       q = final_df[final_df['is_test'] == 1].groupby(grouping_col).size().min()
       ```
   - If `'is_test'` is absent:
     - Default to a sensible value like `q = 7`

     Automatically determine the context length T per group to ensure that:

The number of generated training samples (from the sliding window) is no more than 100 per group

Use the formula:
T=max(1,len(group) - q - max_sliding_samples + 1)
This ensures each group contributes at most 100 training samples while keeping T large enough to allow sliding windows.

Adjust or clip T to a minimum of 1 if necessary.

Apply this logic after splitting into train/test and before generating X_train/y_train.

Build training samples using a **sliding window**:
   - Use the target column's past `T` time steps to predict the next `q` steps
     - For example:
       ```python
       X_train[i] = target[t - T : t]
       y_train[i] = target[t : t + q]
       ```
   - Ensure at least `T + q` time steps are available
   - Skip any group that doesn't have sufficient rows

For testing:
   - Use the **last `T` time steps before the test period** to construct `X_test`
   - Make sure `X_test` covers all groups that have at least `T` + `q` samples
   For each group:

Construct X_test using the last T time steps from the training set — not from the test period.

This means: if using is_test, take the last T rows before the first test row in that group.

If splitting by timestamp, take the last T rows before the 80% cutoff point.

This simulates a real forecasting scenario, where only past data is used to predict future outcomes.
3. **Per group** (based on selected grouping column):
   - Sort by time.
   - Split into train/test by:
     - Prefer `is_test` if the column exists.
     - Else, split by 80% quantile of the timestamp.

4. Return:
   - `X_train_list`: list of arrays (N, T)
   - `y_train_list`: list of arrays (N,) or (N, q)
   - `X_test_list`: list of arrays (1, T)

- Before applying any transformation like imputation or encoding, always check:
  - The DataFrame or array to be transformed is not empty (i.e., has at least one row).
  - The list of columns to transform is not empty.
  - If either condition is not met (empty DataFrame or no columns), skip that group or step entirely to avoid runtime errors.
---

## Inputs

Data sample:
{df_sample}

Project description:
{description}

Code patterns and prior logic:
{patterns}

---

## Output Format

Return {n_candidates} Python code snippet(s). Each must:
- Assign `X_train_list`, `y_train_list`, and `X_test_list`
- Include **only valid Python code**
- Do **not** include markdown, comments, or explanations
- Start each candidate with:

### Candidate i ###
<code here>
"""
    response = call_openai(content)
    # print('RESPONSE', response)
    return response








def split_for_lstm(df, seq_len=30, target_col="sales"):
    df = df.sort_values("date")
    features = df.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    tests = df.get("is_test", pd.Series(0)).values
    X, y = [], []
    arr, targ = features.values, df[target_col].values
    for i in range(len(df) - seq_len):
        if tests[i+seq_len] == 1: continue
        X.append(arr[i:i+seq_len]); y.append(targ[i+seq_len])
    return np.array(X), np.array(y)


def summarize_df(df: pd.DataFrame, name: str) -> str:
    info = f"File: {name}\n"
    info += f"Columns: {', '.join(df.columns)}\n"
    if 'date' in df.columns:
        info += f"Contains time information (date)\n"
    if df.select_dtypes(include='object').shape[1] > 0:
        info += f"Categorical columns: {', '.join(df.select_dtypes(include='object').columns)}\n"
    if df.select_dtypes(include='number').shape[1] > 0:
        info += f"Numerical columns: {', '.join(df.select_dtypes(include='number').columns)}\n"
    return info + "\n"

def summarize_csvs(csv_paths: list[str], nrows: int = 100) -> str:
    summaries = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, nrows=nrows)
            name = os.path.basename(path)
            summaries.append(summarize_df(df, name))
        except Exception as e:
            summaries.append(f"File: {path}\nERROR: {str(e)}\n")
    return "\n".join(summaries)


def infer_grouping_columns_with_llm(final_df):
    """
    Use LLM to return grouping column candidates from a pandas DataFrame.
    Grouping columns must be:
    - categorical (object or category dtype)
    - fewer than 20% unique values
    - have at least one value occurring in >100 rows
    - prefer names like 'store', 'product', 'region', 'series', or 'dept'
    
    Parameters:
    - final_df: pandas DataFrame
    - openai_api_key: (optional) your OpenAI API key
    
    Returns:
    - List of column names (strings) that are valid grouping candidates
    """
    import pandas as pd

    # Extract only schema info for prompt (not data)
    column_info = final_df.dtypes.astype(str).to_dict()
    col_cardinality = {col: final_df[col].nunique() for col in final_df.columns}
    col_counts = {col: final_df[col].value_counts().max() if final_df[col].dtype.name in ['object', 'category'] else None for col in final_df.columns}
    num_rows = len(final_df)

    # Build prompt
    prompt = f"""
You are a helpful data science assistant.

Given a DataFrame's schema and summary stats:

- Data types per column:
{column_info}

- Number of rows:
{num_rows}

- Cardinality (unique values) per column:
{col_cardinality}

- Max frequency (value_counts().max()) per column:
{col_counts}

Your task: return a Python list of **grouping column candidates** for time series forecasting.

Requirements for candidates:

- dtype must be 'object' or 'category'
- cardinality must be **less than 20% of total rows AND no more than 100 unique values**
- max frequency of any value must be **more than 100 rows**
- select at most 2 columns meeting these criteria, prioritizing those matching keywords and lower cardinality

Respond ONLY with a Python list of valid column names, e.g.:
```python
['store_id', 'family']
If no candidates found, return an empty list: [].
"""
    return call_openai(prompt)

def prepare_features_for_model(
    df,
    exclude_cols=None,
    threshold_ratio=0.1,
    max_one_hot=5,
    lag_days=7,
    scale=False,
    scaler_type='standard',  # 'standard' or 'minmax'
    handle_datetime=True,
    verbose=False
):
    df = df.copy()
    if exclude_cols is None:
        exclude_cols = []

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    processed_cols = []
    temp_features = []

    for col in feature_cols:
        col_data = df[col]

        if verbose:
            print(f"Processing column: {col} (dtype: {col_data.dtype})")

        # Handle boolean columns
        if col_data.dtype == bool:
            temp_features.append(pd.Series(col_data.astype(int), name=col))
            processed_cols.append(col)
            continue

        # Handle datetime features
        if handle_datetime and pd.api.types.is_datetime64_any_dtype(col_data):
            temp_features.append(pd.Series(col_data.dt.month, name=f"{col}_month"))
            temp_features.append(pd.Series(col_data.dt.dayofweek, name=f"{col}_weekday"))
            temp_features.append(pd.Series(col_data.dt.hour if hasattr(col_data.dt, 'hour') else 0, name=f"{col}_hour"))
            processed_cols.append(col)
            continue

        # Coerce object to string if needed
        if col_data.dtype == object:
            try:
                col_data = col_data.astype(str)
            except Exception as e:
                warnings.warn(f"Skipping column {col}: cannot convert to string.")
                continue

        col_type = col_data.dtype

        # Categorical
        if col_type == object or col_type.name == 'category':
            unique_vals = col_data.nunique(dropna=True)
            if (unique_vals / len(df)) < threshold_ratio:
                encoded = col_data.fillna("missing").astype(str)
                le = LabelEncoder()
                try:
                    encoded = le.fit_transform(encoded)
                    temp_features.append(pd.Series(encoded, name=col))
                    processed_cols.append(col)
                except Exception as e:
                    warnings.warn(f"Skipping column {col}: Label encoding failed.")
            else:
                if verbose:
                    warnings.warn(f"Dropping column {col}: high cardinality ({unique_vals} unique)")
            continue

        # Integer
        elif np.issubdtype(col_type, np.integer):
            unique_vals = col_data.nunique()
            filled_col = col_data.fillna(col_data.mode().iloc[0] if not col_data.mode().empty else 0)
            if unique_vals == 2:
                temp_features.append(pd.Series(filled_col, name=col))
            elif unique_vals <= max_one_hot:
                one_hot = pd.get_dummies(filled_col, prefix=col)
                temp_features.append(one_hot)
            else:
                freq_encoded = filled_col.map(filled_col.value_counts())
                temp_features.append(pd.Series(freq_encoded, name=f"{col}_freq"))
            processed_cols.append(col)
            continue

        # Float
        elif np.issubdtype(col_type, np.floating):
            filled = col_data.fillna(col_data.mean())
            temp_features.append(pd.Series(filled, name=col))
            # Add lag and percentage change features
            lagdiff = col_data.diff(lag_days).fillna(0)
            pctchange = col_data.pct_change(lag_days).replace([np.inf, -np.inf], 0).fillna(0)
            temp_features.append(pd.Series(lagdiff, name=f"{col}_lagdiff"))
            temp_features.append(pd.Series(pctchange, name=f"{col}_pctchange"))
            processed_cols.append(col)
            continue

        else:
            if verbose:
                warnings.warn(f"Skipping column {col}: unsupported dtype {col_type}")

    # Combine all features efficiently
    new_df = pd.concat(temp_features, axis=1)

    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    new_df = pd.DataFrame(imputer.fit_transform(new_df), columns=new_df.columns, index=df.index)

    # Optional scaling
    if scale:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        new_df = pd.DataFrame(scaler.fit_transform(new_df), columns=new_df.columns, index=df.index)

    if verbose:
        print(f"Final dataset has {new_df.shape[1]} features.")

    return new_df


def generate_time_series_splits(final_df, grouping_cols=None, T=864, q=None, target_cols=None, exogeneous_variable=None):
    final_df = final_df.copy()

    if exogeneous_variable is not None and all(isinstance(x, int) for x in exogeneous_variable):
        exogeneous_variable = [final_df.columns[i] for i in exogeneous_variable]

    if not grouping_cols:
        grouping_cols = []
    elif not isinstance(grouping_cols, list):
        grouping_cols = [grouping_cols]

    # Identify timestamp column
    timestamp_col = None
    for col in final_df.columns:
        if pd.api.types.is_datetime64_any_dtype(final_df[col]):
            timestamp_col = col
            break
    if timestamp_col is None:
        timestamp_col = "synthetic_date"
        final_df[timestamp_col] = pd.date_range(start="2000-01-01", periods=len(final_df), freq="D")

    # Default target columns
    if target_cols is None:
        numeric_cols = [col for col in final_df.columns if final_df[col].dtype in ['int64', 'float64']]
        if not numeric_cols:
            raise ValueError("No numeric columns found to use as target.")
        target_cols = [numeric_cols[0]]
    elif isinstance(target_cols, str):
        target_cols = [target_cols]

    exo_cols = exogeneous_variable if exogeneous_variable is not None else []

    # Output containers
    X_train_list, y_train_list, X_test_list = [], [], []
    y_train_exo_list, y_test_exo_list = [], []
    test_indices_list = []
    group_keys = []
    q_vec = []

    channel_id = None  # Will be computed once and returned

    # Handle grouping
    if grouping_cols:
        grouped = final_df.groupby(grouping_cols)
    else:
        grouped = [(None, final_df)]

    for group_key, df_group in grouped:
        df_group = df_group.sort_values(by=timestamp_col).reset_index(drop=True)
        n = len(df_group)

        # Train/test split
        if 'is_test' in df_group.columns:
            train_df = df_group[df_group['is_test'] == 0]
            test_df = df_group[df_group['is_test'] == 1]
        else:
            split_idx = int(n * 0.8)
            train_df = df_group.iloc[:split_idx]
            test_df = df_group.iloc[split_idx:]

        inferred_q = q if q is not None else test_df.shape[0]
        q_vec.append(inferred_q)
        T_adjusted = T

        if len(train_df) < T_adjusted + inferred_q:
            continue

        # Exogenous variable preprocessing
        if exo_cols:
            train_exo_raw = train_df[exo_cols].copy()
            test_exo_raw = test_df[exo_cols].iloc[:inferred_q].copy()
            common_exo_cols = list(set(train_exo_raw.columns).intersection(set(test_exo_raw.columns)))
            train_exo = train_exo_raw[common_exo_cols].copy()
            test_exo = test_exo_raw[common_exo_cols].copy()

            for col in train_exo.select_dtypes(include='object').columns:
                le = LabelEncoder()
                full_col = pd.concat([train_exo[col], test_exo[col]], axis=0).fillna("missing")
                le.fit(full_col)
                train_exo[col] = le.transform(train_exo[col].fillna("missing"))
                test_exo[col] = le.transform(test_exo[col].fillna("missing"))

            imputer = SimpleImputer(strategy='mean')
            train_exo = pd.DataFrame(imputer.fit_transform(train_exo), columns=common_exo_cols)
            test_exo = pd.DataFrame(imputer.transform(test_exo), columns=common_exo_cols)
        else:
            train_exo = None
            test_exo = None

        # Prepare features
        exclude_cols = [timestamp_col]
        all_model_features = train_df.copy()
        for col in final_df.columns:
            if col not in exclude_cols and col not in all_model_features.columns:
                all_model_features[col] = train_df[col]

        all_test_features = test_df[all_model_features.columns]

        combined_df = pd.concat([all_model_features, all_test_features], axis=0).reset_index(drop=True)
        processed = prepare_features_for_model(combined_df, exclude_cols=[])

        # Get target channel_id once (from processed columns)
        if channel_id is None:
            target_col_name = target_cols[0]
            if target_col_name not in processed.columns:
                raise ValueError(f"Target column '{target_col_name}' not found in processed features.")
            channel_id = list(processed.columns).index(target_col_name)

        # Final train/test features
        train_features = processed.iloc[:len(train_df)].reset_index(drop=True)
        test_features = processed.iloc[len(train_df):].reset_index(drop=True)
        train_targets = train_df[target_cols].reset_index(drop=True)

        # Build training sequences
        X_train, y_train, y_train_exo = [], [], []

        for i in range(min(len(train_features) - T_adjusted - inferred_q + 1, 1000)):
            X_train.append(train_features.iloc[i:i+T_adjusted].values)                  # shape: (T, D_all)
            y_train.append(train_features.iloc[i+T_adjusted:i+T_adjusted+inferred_q].values)                   # shape: (T, D_targets)
            if train_exo is not None:
                y_train_exo.append(train_exo.iloc[i+T_adjusted:i+T_adjusted+inferred_q].values)

        X_train = np.array(X_train)            # (B, T, D_all)
        y_train = np.array(y_train)            # (B, T, D_targets)
        y_train_exo = np.array(y_train_exo) if y_train_exo else None

        # Prepare test set
        if len(test_df) >= inferred_q:
            X_test = train_features.iloc[-T_adjusted:].values.reshape(1, T_adjusted, -1)
            test_exo_np = test_exo.iloc[:inferred_q].values.reshape(1, inferred_q, -1) if test_exo is not None else None
            test_indices = test_df["id"].iloc[:inferred_q].tolist() if "id" in test_df.columns else [None] * inferred_q
            test_indices_list.append(test_indices)
            group_keys.append(group_key)
        else:
            X_test = None
            test_exo_np = None

        if len(X_train) > 0 and len(y_train) > 0 and X_test is not None:
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            X_test_list.append(X_test)
            y_train_exo_list.append(y_train_exo)
            y_test_exo_list.append(test_exo_np)

    return X_train_list, y_train_list, X_test_list, y_train_exo_list, y_test_exo_list, test_indices_list, group_keys, q_vec, channel_id             
    






def clean_llm_list_output(llm_output: str):
    # Remove markdown code block markers
    if llm_output.startswith("```python"):
        llm_output = llm_output[len("```python"):].strip()
    if llm_output.endswith("```"):
        llm_output = llm_output[:-3].strip()
    # Now safely parse the list string
    try:
        parsed_list = ast.literal_eval(llm_output)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError("Parsed output is not a list")
    except Exception as e:
        raise ValueError(f"Failed to parse LLM output as list: {e}")
    

def merge_process(schemas, project_description, dfs, pattern_list):
    try:
        # raw_merge_code = synthesize_merge_code(schemas, pattern_list, project_description, n_candidates=1)
        # print('CODE FOR MERGING', raw_merge_code)
        raw_merge_code = PREDEFINED_CODE
        merge_candidates = split_candidates(raw_merge_code)
        cleaned_code = clean_code(merge_candidates[0])
        # for i in range(len(dfs)):
        #     print(dfs[i].head)

        temp_ns = dict(dfs)
        temp_ns.update({'pd': pd, 'np': np, 'OneHotEncoder': OneHotEncoder})
        exec_code_with_imports(cleaned_code, temp_ns)

        merged_df = temp_ns['final_df']
        return merged_df, temp_ns, cleaned_code, None
    except Exception as e:
        return None, None, None, str(e)