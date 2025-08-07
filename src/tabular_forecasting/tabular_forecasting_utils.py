import os
import json
import pandas as pd
import traceback
# import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
from openai import OpenAI
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import warnings
import streamlit as st

# Disable all warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
import lightgbm as lgb

def train_and_predict(X_train, y_train, X_test, X_val=None, y_val=None):
    # Remove unwanted columns
    exclude_cols = {'date', 'id'}
    common_cols = list(set(X_train.columns).intersection(X_test.columns) - exclude_cols)
    common_cols.sort()

    # If validation sets are not provided, split into training and validation sets
    if X_val is None or y_val is None:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train[common_cols], y_train, test_size=0.2, random_state=42
        )
    else:
        X_tr, y_tr = X_train[common_cols], y_train

    # LightGBM Dataset
    lgb_train = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    # LightGBM parameters
    params = {
        'objective': 'regression',
        'random_state': 42
    }

    # Train the model with early stopping
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_val],
        num_boost_round=1000,
        early_stopping_rounds=10
    )

    # Validation prediction
    y_pred_val = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_pred_val)

    # Predict test
    pred_test = model.predict(X_test[common_cols])

    return pred_test, val_mse



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

def download_kernels(api, kernels, target_dir="kaggle_notebooks"):
    os.makedirs(target_dir, exist_ok=True)
    paths = []

    for k in kernels:
        kernel_ref = getattr(k, 'ref', None)

        if not kernel_ref:
            print(f"Skipping kernel with missing ref for user {getattr(k, 'author', 'Unknown')}")
            continue

        try:
            print(f"Pulling kernel: {kernel_ref}")
            path = api.kernels_pull(kernel_ref, path=target_dir)
            paths.append(path)
        except Exception as e:
            print(f"Error pulling {kernel_ref}: {e}")

    return paths

def extract_preprocessing_patterns_from_notebook(nb_content: str) -> str:
    prompt = f"""
You are given the content of a Jupyter notebook, which may contain Pandas preprocessing logic.
Your job is to extract the PREPROCESSING patterns and hints that may be helpful.

Return ONLY a JSON list of short preprocessing code snippets (1-5 lines each), such as encoding, filling NaNs, creating time features, removing duplicates, creating innovative features, etc.

Notebook content:
{nb_content}

Return JSON list of patterns and preprocessing hints only.
"""
    return call_openai(prompt)

def extract_merge_patterns_from_notebook(nb_content: str) -> str:
    prompt = f"""
You are given the content of a Jupyter notebook, which may contain merge hints logic.
Your job is to extract the MERGE patterns of csv and hints that may be helpful.

Return ONLY a JSON list of short merging logic code snippets (1-5 lines each), such as join, merge remove some csv, important csvs, ...

Notebook content:
{nb_content}

Return JSON list of merges hints only.
"""
    return call_openai(prompt)


def extract_preprocessing_patterns_from_kaggle(api, competition='store-sales-time-series-forecasting', page_size=5):
    kernels = api.kernels_list(competition=competition, page_size=page_size)
    paths = download_kernels(api, kernels)
    pattern_list = []
    for path in paths:
        try:
            nb_content = open(path).read()
            # print('Content preproc', nb_content)
            extracted = extract_preprocessing_patterns_from_notebook(nb_content)
            # print('Extracted preproc', extracted)
            pattern_list.append(extracted)
            # print('pattern_list', pattern_list)
        except Exception as e:
            print(f"Error extracting preprocessing from {path}: {e}")
    return pattern_list


def extract_merge_patterns_from_kaggle(api, competition='store-sales-time-series-forecasting', page_size=1):
    kernels = api.kernels_list(competition=competition, sort_by='voteCount', page_size=page_size)
    paths = download_kernels(api, kernels)
    pattern_list = []
    for path in paths:
        try:
            nb_content = open(path).read()
            # print('content merge', nb_content)
            extracted = extract_merge_patterns_from_notebook(nb_content)
            # print('extracted merge', extracted)
            pattern_list.append(extracted)
            # print('pattern_list', pattern_list)
        except Exception as e:
            print(f"Error parsing notebook {path}: {e}")
    return pattern_list

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

# --- OpenAI Client Setup ---
client = OpenAI(
    base_url='http://api.openai.ukrc.huawei.com:4000/v1',
    api_key='sk-1234',
)

def call_openai(content, model='qwen2.5-coder-32b-instruct', temperature=0.3, max_tokens=2048):
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
    lines = code_str.splitlines()
    lines = [line for line in lines if line.strip() not in ("```", "```python", "```python3")]
    return "\n".join(lines)

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


def evaluate_merge_with_preprocessing(merge_code, preprocess_code, dfs, target_col='sales'):
    namespace = dict(dfs)
    # Pre-inject common libs for safety
    namespace.update({
        'pd': pd,
        'np': np,
        'OneHotEncoder': OneHotEncoder
    })
    try:
        exec_code_with_imports(clean_code(merge_code), namespace)

        if 'final_df' not in namespace:
            print("Missing final_df")
            return None
        namespace['final_df'] = namespace['final_df'].copy()
        # print('1', preprocess_code)
        # print('2', clean_code_preprocess(preprocess_code))
        exec_code_with_imports(clean_code_preprocess(preprocess_code), namespace)

        df = namespace.get("processed_df")
        if df is None or target_col not in df.columns:
            print(preprocess_code)
            print("Invalid processed_df")
            return None

        X = df.drop(columns=[target_col])
        y = df[target_col]

        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].astype('int64') // 10**9

        X = pd.get_dummies(handle_missing_nan(make_unique_columns(clean_feature_names(X))), drop_first=True)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        train_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_val, y_val, reference=train_data)

        model = lgb.train({
            'objective': 'regression', 'metric': 'mse', 'verbosity': -1, 'seed': 42
        }, train_data, valid_sets=[train_data, val_data])

        preds = model.predict(X_val)
        return mean_squared_error(y_val, preds)

    except Exception as e:
        traceback.print_exc()
        return None

def handle_missing_nan(X):
    X_clean = X.copy()

    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    for column in X_clean.columns:
        if pd.api.types.is_numeric_dtype(X_clean[column]):
            # For numeric columns, replace NaN with the median or 0
            median = X_clean[column].median()
            X_clean[column] = X_clean[column].fillna(0 if np.isnan(median) else median)
        else:
            # For non-numeric columns, try to label encode and then impute
            try:
                # Label encode the column
                X_clean[column] = label_encoder.fit_transform(X_clean[column].astype(str))

                # Replace NaN with the median or 0
                median = X_clean[column].median()
                X_clean[column] = X_clean[column].fillna(0 if np.isnan(median) else median)
            except:
                # If label encoding fails, discard the column
                X_clean.drop(column, axis=1, inplace=True)

    return X_clean

def make_unique_columns(df):
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for col in cols:
        if col not in seen:
            seen[col] = 0
            new_cols.append(col)
        else:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
    df.columns = new_cols
    return df

def clean_feature_names(df):
    df.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in df.columns]
    return df

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

def synthesize_merge_code(schemas, patterns, description, n_candidates=1):
    content = f"""
        You are a Pandas expert. You have these pandas DataFrames already loaded, with variable names matching the filenames without extensions (e.g., train, test, stores, transactions, oil, holidays):

        {schemas}

        Based on actual Kaggle merge patterns:

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
    return call_openai(content)


def synthesize_preprocessing_code(df_summary, description, patterns, n_candidates=1):
    content = f"""
        You are a data preprocessing expert. You are working on a pandas DataFrame called `final_df`, which was already produced by merging multiple CSVs.

        DO NOT read any CSV files. DO NOT use pd.read_csv or any file I/O.

        Use only the already-loaded DataFrame `final_df`.

        Here is the summary of `final_df`:
        {df_summary}

        Below are examples of real preprocessing code patterns from Kaggle notebooks:
        {patterns}

        Goal of the project:
        {description}

        Write {n_candidates} distinct Pandas preprocessing strategies to transform `final_df` into `processed_df`, which will be used for modeling.

        You can:
        - Engineer new features (e.g. from 'date', lags, rolling stats)
        - Encode categoricals (Prefer LabelEncoder than OneHotEncoder and if needed use OneHotEncoder with `sparse_output=False` — do NOT use `sparse=`)
        - Impute missing values
        - Remove duplicates
        - Scale or normalize features
        - Create time-based flags
        - Ensure no data leakage
        - If the input `final_df` contains an `id` column, DO NOT drop it in `processed_df`, even after preprocessing.
        - If you drop `id` temporarily, make sure to keep a copy and reattach it to `processed_df` at the end.
        - If there is no `id`, preserve a combination of columns that could serve as a row identifier and keep is_test column that explain if yes or no the row is part of the train or test
        - Check for existence of columns to avoid errors like ['type'] not in index". The code should work without any error.
        ⚠️ Important:
        - - Check for existence of columns to avoid errors like ['type', 'family', ...] not in index. CHECK everything". The code should work without any error.
        - INCLUDE all necessary import statements (like pandas, numpy, and OneHotEncoder).
        - If you use `OneHotEncoder`, use: `OneHotEncoder(sparse_output=False)`
        - Do not use `sparse=True` or `sparse=False`, as they may raise errors.
        ⚠️ Ensure that all inputs to encoders (e.g., LabelEncoder, OneHotEncoder) are explicitly converted to strings or numbers before fitting.

        Return valid Python code only. Each strategy must assign the final output to a variable called `processed_df`.

        Separate each strategy with this line: ### Candidate n ###
        """
    return call_openai(content)


def strip_code_fences(code: str) -> str:
    """
    Strips ```python ... ``` or ``` ... ``` fences from LLM output.
    """
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?", "", code).strip()
    if code.endswith("```"):
        code = re.sub(r"```$", "", code).strip()
    return code


def inverse_transform_predictions(preprocess_code: str, y_pred: np.ndarray) -> np.ndarray:
    """
    Automatically infers and applies inverse transformations to y_pred
    based on the preprocessing code.

    Parameters:
    - preprocess_code (str): The original code string that includes preprocessing logic
    - y_pred (np.ndarray): The model prediction (possibly transformed)

    Returns:
    - np.ndarray: Inverse-transformed predictions (clipped if appropriate)
    """
    prompt = f"""
You are a data preprocessing expert.

The user trained a model on a target variable y that was transformed during preprocessing.
Now, the user wants to inverse-transform the predictions from the model.

Here is the preprocessing code:
{preprocess_code}

Please analyze this code and return only a Python function named `inverse_y_transform(y_pred)`.

The function should:
- Apply the correct inverse transformation to y_pred (e.g., np.expm1 if np.log1p was used)
- Assume placeholder values (mean, std, min, max) if needed for scaling reversal
- Clip the final result to be >= 0 if the target is a forecast like sales
- Be self-contained and valid Python — do NOT include markdown (no ```)

Return ONLY the function. No explanation, no comments, no markdown.
"""

    result = call_openai(prompt)

    # Strip markdown code fences if accidentally returned
    result_clean = strip_code_fences(result)

    namespace = {}
    try:
        exec(result_clean, {'np': np}, namespace)
        inverse_fn = namespace.get("inverse_y_transform")
        if not inverse_fn:
            raise ValueError("No function `inverse_y_transform` defined.")
        return inverse_fn(y_pred)
    except Exception as e:
        print("⚠️ Failed to execute inverse transform logic:", e)
        print("Original returned code:\n", result_clean)
        return y_pred  # fallback

def summarize_final_df(df: pd.DataFrame) -> str:
    return f"Shape: {df.shape}\nColumns: {', '.join(df.columns)}\nCategoricals: {', '.join(df.select_dtypes(include='object').columns)}\nNumericals: {', '.join(df.select_dtypes(include='number').columns)}"
def strip_code_fences(code: str) -> str:
    """
    Strips ```python ... ``` or ``` ... ``` fences from LLM output.
    """
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python)?", "", code).strip()
    if code.endswith("```"):
        code = re.sub(r"```$", "", code).strip()
    return code



def train_tabular(X_train, y_train, X_test):

    # Initial data cleaning
    nan_threshold = 0.5
    nan_percentages = X_test.isna().mean()
    columns_to_drop = nan_percentages[nan_percentages > nan_threshold].index
    X_test1 = X_test.drop(columns=columns_to_drop)
    X_train1 = X_train.drop(columns=columns_to_drop)
    # Remove unwanted columns
    exclude_cols = {'date', 'id'}
    common_cols = list(set(X_train1.columns).intersection(X_test1.columns) - exclude_cols)
    common_cols.sort()
    X_train = X_train1[common_cols]
    X_test = X_test1[common_cols]

    st.write(f"✅ Dropped {len(columns_to_drop)} columns with >{nan_threshold*100:.0f}% NaNs")

    # Ensure categoricals are treated right
    for df in [X_train, X_test]:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].astype('category')

    # Split train/validation (no shuffle for time series)
    training_size = int(0.8 * X_train.shape[0])
    X_tr, X_val = X_train[:training_size], X_train[training_size:]
    y_tr, y_val = y_train[:training_size], y_train[training_size:]

    if 'orig_index' in X_tr.columns:
        X_tr = X_tr.drop(columns=['orig_index'])
    if 'orig_index' in X_val.columns:
        X_val = X_val.drop(columns=['orig_index'])

    # Train LightGBM
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

    # evals_result = {}
    with st.spinner("🏋️ Training LightGBM Model..."):
        model = lgb.LGBMRegressor(**hyper_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            eval_metric='l1',
        )

    # Collect eval metric curve
    # l1_values = evals_result['valid_0']['l1']

    # Make predictions
    X_test_features = X_test.drop(columns=['orig_index']) if 'orig_index' in X_test.columns else X_test.copy()
    y_pred = np.clip(model.predict(X_test_features), 0, None)
    y_val_pred = np.clip(model.predict(X_val), 0, None)
    y_tr_pred = np.clip(model.predict(X_tr), 0, None)

    return y_pred, y_val, y_val_pred, y_tr_pred
