import re
from typing import Optional, Any
import numpy as np
from openai import OpenAI
import torch
import base64
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
import inspect
from src.reinforcement_learning import (ContextualBanditSuccessiveRejects, ContextualBanditUniformBAI, ContextualBanditSuccessiveHalving,
                                        ContextualBanditLUCB, genetic_algorithm_reinforcement, contextual_bandit_reinforcement,
                                        contextual_bandit_random)



def analyze_forecast_and_plot(y_true, y_pred, output_dir="plots", id="example"):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from scipy.signal import find_peaks

    os.makedirs(output_dir, exist_ok=True)

    # Flatten if needed (e.g., for [T, 1])
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    # Compute error metrics
    residual = y_true - y_pred
    mse = np.mean(residual**2)
    mae = np.mean(np.abs(residual))
    bias = np.mean(residual)

    if len(y_true) >= 2 and len(y_pred) >= 2:
        trend_diff = np.mean(np.gradient(y_true)) - np.mean(np.gradient(y_pred))
    else:
        trend_diff = float('nan')  # or 0.0 or skip this metric

    try:
        lag = np.argmax(np.correlate(y_true, y_pred, mode='full')) - len(y_true) + 1
    except Exception:
        lag = float('nan')

    try:
        peaks_true, _ = find_peaks(y_true)
        peaks_pred, _ = find_peaks(y_pred)
        peak_miss_rate = 1 - len(set(peaks_true).intersection(peaks_pred)) / (len(peaks_true) + 1e-5)
    except Exception:
        peak_miss_rate = float('nan')

    stats = {
        "MSE": round(mse, 3),
        "MAE": round(mae, 3),
        "Bias": round(bias, 3),
        "Trend Difference": round(trend_diff, 3) if not np.isnan(trend_diff) else "N/A",
        "Lag": lag if not np.isnan(lag) else "N/A",
        "Peak Miss Rate": round(peak_miss_rate, 3) if not np.isnan(peak_miss_rate) else "N/A",
    }

    # Save plot
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Ground Truth", color="black")
    plt.plot(y_pred, label="Prediction", color="orange", linestyle="--")
    plt.title("Prediction vs Ground Truth")
    plt.legend()
    filepath = os.path.join(output_dir, f"forecast_{id}.png")
    plt.savefig(filepath)
    plt.close()

    return stats, filepath

def extract_apply_method(generated_class_code):
    """
    Extracts the `apply` method from the generated class code.
    This method will capture the entire logic of the `apply` function.
    """
    # Regular expression to match the apply method and its contents
    apply_method_pattern = re.compile(r'def\s+apply\(.*?\):\s*(.*?)\s*else\s+raise', re.DOTALL)
    match = apply_method_pattern.search(generated_class_code)

    if match:
        # Return the captured content inside the apply method (i.e., the method body)
        return match.group(1)
    return ''  # Return empty string if no match is found


# # Function to build the prompt for the model based on user feedback
def build_feedback_prompt(feedback, stats_str, img_path):
    """
    Build a prompt for the model to generate Python class and parameters based on the feedback.
    The prompt ensures only the class and the function are generated and are easy to extract.
    Additionally, it ensures that the `function_type` in the class is the same as the `action` in the `generate_random_params_for_action` function.
    """
    return f"""
    You are an expert machine learning assistant. Your task is to analyze the performance of a time series forecasting model and generate transformation logic in Python to correct or adapt the predictions.

    The following diagnostic information is provided:
    **Meta data of the project and expert information**:
    {feedback}

    📊 **Forecast Error Statistics**:
    {stats_str}
    Given the following feedback about a time series prediction model:

    Feedback: "{feedback}"

    Please generate a Python class called `GenericFunction` that represents a transformation function based on the feedback. The class should include the following:

    1. A constructor (`__init__`) that accepts two parameters:
       - `function_type`: The type of transformation (the one put by the user). This should match the `action` parameter used in the `generate_random_params_for_action` function.
       - `params`: A dictionary containing parameters specific to the transformation.

    2. An `apply` method that takes two arguments:
       - (a) a time series (prediction) which should be modified of size N (sample size), T(time step), D(dimension)
       - (b) the context vector (batch_x) of size N(sample size), T(time step), D (dimension)
       The method should apply the transformation inferred from the user's feedback either on the prediction, the context vector, or both and output only the time series after prediction (only one output).
       The output of the transfrmation should be the same type, dtype and shape as the predictions in input.

    Additionally, please generate a function called `generate_random_params_for_action` that takes two parameters:
       - `action`: The type of transformation (the one put by the user). This should be the same as the `function_type` defined in the class.
       - `batch_x`: the time series of the context vector. Note that batch_x is a tensor of size N, T, D

    The function should return a dictionary of parameters based on the user's feedback. This dictionary cannot be empty.

    Please ensure that the generated class and function are valid Python code and clearly separated. The class and the function should be output as follows:

    --- START OF GENERATED CODE ---

    # Class Definition:
    class GenericFunction:
    <class-body>

    # Function Definition:
    def generate_random_params_for_action(action, batch_x):
    <function-body>

    --- END OF GENERATED CODE ---

    Please ensure that the generated code is valid, formatted correctly, and that the `function_type` in the `GenericFunction` class matches the `action` in the `generate_random_params_for_action` function.
    """
# def build_feedback_prompt(feedback, stats, image_path):
#     """
#     Build a prompt for a multimodal LLM to generate Python code based on statistical feedback and a visualization.
#     """

#     stats_str = "\n".join([f"- {k}: {v}" for k, v in stats.items()])

#     return f"""
#         You are an expert machine learning assistant. Your task is to analyze the performance of a time series forecasting model and generate transformation logic in Python to correct or adapt the predictions.

#         The following diagnostic information is provided:
#         **Meta data of the project and expert information**:
#         {feedback}

#         📊 **Forecast Error Statistics**:
#         {stats_str}

#         🖼️ **Attached Image Description**:
#         The image provided shows **model predictions versus ground truth** on the **validation set**. It visually illustrates systematic **biases or distortions** in the prediction behavior of the model.

#         ---

#         Using both the statistics and the image, generate a Python class named `GenericFunction` that represents some candidates transformation function to correct for bias. The class should include:

#         ### 1. Constructor (`__init__`)
#         - `function_type`: a string describing the type of transformation (inferred or user-specified).
#         - `params`: a dictionary of transformation-specific parameters.

#         ### 2. Method: `apply(prediction, batch_x)`
#         - Inputs:
#             - `prediction`: a time series tensor of shape [N, T, D]
#             - `batch_x`: a context tensor of the same shape
#         - Output:
#             - a transformed prediction of the **same shape and dtype** as the input. Please ensure the size of the output is [N, T, D]

#         ### 3. Function: `generate_random_params_for_action(action, batch_x)`
#         - Inputs:
#             - `action`: the transformation name
#             - `batch_x`: tensor of shape [N, T, D]
#         - Output:
#             - a dictionary with randomly sampled parameter values relevant to the action and context.

#         Ensure the output is clean, commented, and ready to integrate.

#         Please format your response **exactly** like this:

#         --- START OF GENERATED CODE ---

#         # Class Definition:
#         class GenericFunction:
#             <class-body>

#         # Function Definition:
#         def generate_random_params_for_action(action, batch_x):
#             <function-body>

#         --- END OF GENERATED CODE ---
#         """


def get_generated_code(feedback, stats, image_path):
    client = OpenAI(
    base_url='http://api.openai.ukrc.huawei.com:4000/v1',  # Update to your base URL
    api_key='sk-1234',
)
    prompt = build_feedback_prompt(feedback, stats, image_path)
    # Load image and encode to base64
    # image_path = "/home/mtiomoko/post_training_forecasting_official-main/plots/prediction_vs_truth.png"
    # with open(image_path, "rb") as f:
    #     image_base64 = base64.b64encode(f.read()).decode("utf-8")

    # # Convert to data URI
    # image_data_uri = f"data:image/jpeg;base64,{image_base64}"

    # Chat completion call with image input
    # response = client.chat.completions.create(
    #     model="internvl2.5-78b",
    #     messages=[
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": prompt},
    #                 {"type": "image_url", "image_url": {"url": image_data_uri}}
    #             ]
    #         }
    #     ],
    #     temperature=0.5,
    #     max_tokens=10000,
    # )
    # Call OpenAI API to generate the code
    response = client.chat.completions.create(
        model="qwen2.5-coder-32b-instruct",  # Update to the appropriate model
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.5,  # Adjust temperature for creativity level
        max_tokens=10000,
    )

    # Extract and return the generated code
    return response.choices[0].message.content


def get_generic_function_class(generated_code):
    """
    Extracts the GenericFunction class from the generated code.
    Returns the class definition or None if the class is not found.
    """
    print(generated_code)  # Check the generated code

    # Regex to match 'class GenericFunction' and everything until the next marker
    class_pattern = re.compile(r'# Class Definition:.*?(class\s+GenericFunction.*?)(?=\n# Function Definition:)', re.DOTALL)
    match = class_pattern.search(generated_code)

    if match:
        return match.group(1)  # Returns the class definition including its body

    return None

def get_generate_random_parameters_function(generated_code):
    """
    Extracts the generate_random_params_for_action function from the generated code.
    Returns the function definition or None if the function is not found.
    """
    # Regex to extract everything between # Function Definition: and --- END OF GENERATED CODE ---
    # Allow for any content between the markers, including imports or other code.
    function_pattern = re.compile(r'# Function Definition:.*?(def\s+generate_random_params_for_action.*?)(?=\n--- END OF GENERATED CODE ---)', re.DOTALL)
    match = function_pattern.search(generated_code)

    # If no match is found, return None.
    if match:
        return match.group(1)  # Returns the function definition including its body

    return None  # Return None if the function is not found

def get_function_types_from_class(class_code):
    """
    This function extracts the transformation types from the class's `apply` method logic.
    We look for 'if self.function_type == <function_name>' statements in the code.
    """
    function_types = []

    # Regular expression to match function types after `if self.function_type ==`
    pattern = r"if\s+self\.function_type\s*==\s*['\"]([^'\"]+)['\"]"
    matches = re.findall(pattern, class_code)

    # Add all unique matches to the function types list
    function_types.extend(matches)

    return function_types
import re
def sanitize_generated_code(code):
    """Sanitize the generated code to remove unwanted characters like backticks or any syntax issues."""
    if code:
        # Remove backticks or unwanted characters (e.g., from markdown formatting)
        sanitized_code = code.replace('```', '').strip()
        return sanitized_code
    return code

def execute_generated_code(code):
    """Executes generated code while ensuring imports are handled separately."""
    sanitized_code = sanitize_generated_code(code)
    if not sanitized_code:
        return  # Avoid executing empty or invalid code blocks

    # Split code into lines
    lines = sanitized_code.split("\n")

    # Filter out import statements (they were already executed separately)
    non_import_code = "\n".join([line for line in lines if not line.strip().startswith(("import", "from"))])

    # Execute the remaining code
    exec(non_import_code, globals())
def extract_import_statements(*code_blocks):
    """Extracts import statements from multiple code blocks and returns them as a single string."""
    import_lines = set()  # Use a set to avoid duplicate imports
    for code in code_blocks:
        if code:
            lines = code.split("\n")
            for line in lines:
                if line.strip().startswith("import") or line.strip().startswith("from"):
                    import_lines.add(line.strip())  # Store unique import lines
    return "\n".join(import_lines)

def safe_exec(code_str, label, retries=5):
    for attempt in range(retries + 1):
        try:
            exec(code_str, globals())
            return True
        except Exception as e:
            print(f"[Attempt {attempt+1}] Error in {label}: {type(e).__name__}: {e}")
            if attempt < retries:
                error_prompt = (
                    f"The following generated code for `{label}` failed with error:\n"
                    f"{type(e).__name__}: {e}\n\n"
                    f"Please correct the code:\n\n{code_str}"
                )
                code_str = get_generated_code(error_prompt, None, None)  # stats/image_path might be passed in
            else:
                return False
    return False

def safe_run_rl_algorithm(
    rl_algorithm_fn,
    exp, args,
    new_generic_function_class,
    generate_random_params_function,
    function_types,
    pred, true, batch_x,
    stats, image_path,
    max_retries=5
):
    last_error = None
    for attempt in range(max_retries):
        try:
            return rl_algorithm_fn(
                exp=exp, args=args, episodes=10, alpha=0, gamma=0, epsilon=0,
                action_budget=20, improvement_threshold=0.0,
                generic_function_class=new_generic_function_class,
                generate_random_parameters_function=generate_random_params_function,
                function_types=function_types,
                pred=pred, true=true, batch_x=batch_x,
                streamlit=True, feedback=True
            )
        except Exception as e:
            last_error = e
            print(f"[RL Attempt {attempt+1}] Error: {type(e).__name__}: {e}")

            if attempt < max_retries - 1:
                error_prompt = (
                    f"The generated code caused the RL algorithm to fail with the following error:\n"
                    f"{type(e).__name__}: {e}\n\n"
                    f"Please correct the code:\n\n{inspect.getsource(rl_algorithm_fn)}"
                )
                corrected_code = get_generated_code(error_prompt, stats, image_path)

                # Re-execute corrected components
                Generic_class_code = get_generic_function_class(corrected_code)
                random_params_code = get_generate_random_parameters_function(corrected_code)
                imports_code = extract_import_statements(Generic_class_code, random_params_code)

                for label, code in [
                    ("imports", imports_code),
                    ("GenericFunction class", Generic_class_code),
                    ("generate_random_params_for_action", random_params_code)
                ]:
                    if not safe_exec(code, label):
                        raise RuntimeError(f"Failed to re-execute corrected: {label}")

                new_generic_function_class = globals().get('GenericFunction')
                generate_random_params_function = globals().get('generate_random_params_for_action')
                function_types = get_function_types_from_class(corrected_code)
            else:
                raise RuntimeError(f"RL algorithm failed after {max_retries} attempts: {last_error}")
def handle_feedback(
    exp: Any,
    args: Any,
    pred: Optional[np.ndarray] = None,
    true: Optional[np.ndarray] = None,
    batch_x: Optional[np.ndarray] = None,
    streamlit: bool = False,
    method: str = 'random',
    feedback_text: str = None,
):
    if feedback_text is None:
        project_desc = getattr(args, 'project_description', '')
        stats_summary = getattr(args, 'stats', '')
        feedback_text = f"Improve preprocessing and modeling using this project description:\n{project_desc}\n\nStats:\n{stats_summary}"

    stats, image_path = analyze_forecast_and_plot(true, pred)
    generated_code = get_generated_code(feedback_text, stats, image_path)

    def safe_exec(code_str, label, retries=5):
        for attempt in range(retries + 1):
            try:
                exec(code_str, globals())  # Just checks for syntax/runtime errors at import
                return True
            except Exception as e:
                print(f"[Attempt {attempt+1}] Error in {label}: {type(e).__name__}: {e}")
                if attempt < retries:
                    error_prompt = (
                        f"The following generated code for `{label}` failed with error:\n"
                        f"{type(e).__name__}: {e}\n\n"
                        f"Please correct the code:\n\n{code_str}"
                    )
                    code_str = get_generated_code(error_prompt, stats, image_path)
                else:
                    return False
        return False

    Generic_class_code = get_generic_function_class(generated_code)
    random_params_code = get_generate_random_parameters_function(generated_code)
    imports_code = extract_import_statements(Generic_class_code, random_params_code)

    if not safe_exec(imports_code, "imports"):
        raise RuntimeError("Failed to execute import statements.")
    if not safe_exec(Generic_class_code, "GenericFunction class"):
        raise RuntimeError("Failed to define GenericFunction.")
    if not safe_exec(random_params_code, "generate_random_params_for_action"):
        raise RuntimeError("Failed to define random params function.")

    new_generic_function_class = globals().get('GenericFunction')
    generate_random_params_function = globals().get('generate_random_params_for_action')

    if new_generic_function_class is None:
        raise ValueError("The GenericFunction class was not defined correctly.")
    if generate_random_params_function is None:
        raise ValueError("The generate_random_params_for_action function was not defined correctly.")

    function_types = get_function_types_from_class(generated_code)

    N_ITERATIONS = 15
    MAX_ITER_HYPEROPT = 10

    if method == 'random':
        rl_algorithm = contextual_bandit_random
    elif method == "SR-HPO":
        rl_algorithm = ContextualBanditSuccessiveRejects(
            n_function_types=len(function_types), n_iterations=N_ITERATIONS,
            max_iter_hyperopt=MAX_ITER_HYPEROPT, n_jobs=args.n_jobs)
    elif method == "U-HPO":
        rl_algorithm = ContextualBanditUniformBAI(
            n_function_types=len(function_types), n_iterations=N_ITERATIONS,
            max_iter_hyperopt=MAX_ITER_HYPEROPT, n_jobs=args.n_jobs)
    elif method == "SH-HPO":
        rl_algorithm = ContextualBanditSuccessiveHalving(
            n_function_types=len(function_types), n_iterations=N_ITERATIONS,
            max_iter_hyperopt=MAX_ITER_HYPEROPT, n_jobs=args.n_jobs)
    elif method == "LUCB-HPO":
        rl_algorithm = ContextualBanditLUCB(
            n_function_types=len(function_types), n_iterations=N_ITERATIONS,
            max_iter_hyperopt=MAX_ITER_HYPEROPT, n_jobs=args.n_jobs)
    elif method == 'Genetic':
        rl_algorithm = genetic_algorithm_reinforcement
    elif method == 'PPO':
        rl_algorithm = contextual_bandit_reinforcement
    else:
        raise ValueError(f"Unsupported method: '{method}'")

    # function_set_per_channel, best_mse, best_pred, true, batch_x = rl_algorithm(
    #     exp=exp, args=args, episodes=10, alpha=0, gamma=0, epsilon=0,
    #     action_budget=20, improvement_threshold=0.0,
    #     generic_function_class=new_generic_function_class,
    #     generate_random_parameters_function=generate_random_params_function,
    #     function_types=function_types,
    #     pred=pred, true=true, batch_x=batch_x,
    #     streamlit=True, feedback=True
    # )
    function_set_per_channel, best_mse, best_pred, true, batch_x = safe_run_rl_algorithm(
    rl_algorithm,
    exp, args,
    new_generic_function_class,
    generate_random_params_function,
    function_types,
    pred, true, batch_x,
    stats, image_path,
    max_retries=5
)

    return new_generic_function_class, best_pred, best_mse, true, function_set_per_channel




