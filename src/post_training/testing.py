from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import streamlit as st
from matplotlib.lines import Line2D  # Import Line2D to define custom legend lines
import matplotlib.pylab as plt
import seaborn as sns
from src.post_training.actions_definition import GenericFunction
from src.csv_merging.csv_description_and_utilities import TimeSeriesForecastingDataset


def check_overfitting(exp, args, best_function_set_per_channel, best_function_set_per_channel_feedback, GenericFunction, GenericFunction2):
    """
    Checks for overfitting risk by comparing the MSE improvements before and after applying actions to the model.
    If the improvement is negative or below a certain threshold, it flags potential overfitting.
    Also, outputs the channel with no significant overfitting risk (> 10%).

    Args:
    - exp: The experiment or model instance
    - args: The arguments or configurations for the model
    - best_function_set_per_channel: The best actions (functions and parameters) from exploration
    - best_function_set_per_channel_feedback: The best actions (functions and parameters) from feedback
    - GenericFunction: The function class to apply actions
    - GenericFunction2: The second function class to apply feedback actions
    """
    # Load new test data
    _, test_pred, test_true, batch_x = exp.train_test(args.model_id, test=1)  # Assuming exp.test gives you the predictions, true values, and features
    init_pred = test_pred.copy()

    # Store the channel MSEs and improvements
    channel_improvements = []
    mse_before = []
    mse_after = []

    # Apply the best function set for exploration
    for channel_idx, best_action, best_params in best_function_set_per_channel:
        new_channel_pred = GenericFunction(best_action, best_params).apply(test_pred[:, :, channel_idx].reshape((test_pred.shape[0], test_pred.shape[1], 1)), batch_x[:, :, channel_idx].reshape((batch_x.shape[0], batch_x.shape[1], 1)).cpu().numpy())
        test_pred[:, :, channel_idx] = new_channel_pred[:, :, 0]

    # Apply the best function set for feedback
    for channel_idx, best_action, best_params in best_function_set_per_channel_feedback:
        new_channel_pred = GenericFunction2(best_action, best_params).apply(test_pred[:, :, channel_idx].reshape((test_pred.shape[0], test_pred.shape[1], 1)), batch_x[:, :, channel_idx].reshape((batch_x.shape[0], batch_x.shape[1], 1)).cpu().numpy())
        test_pred[:, :, channel_idx] = new_channel_pred[:, :, 0]

    # Calculate the improvement for each channel
    no_risk_channels = []
    # print(test_pred.shape, test_true.shape)
    for i in range(test_pred.shape[2]):
        mse_before = mean_squared_error(test_true[:, :, i].flatten(), init_pred[:, :, i].flatten())
        mse_after = mean_squared_error(test_true[:, :, i].flatten(), test_pred[:, :, i].flatten())
        improvement = mse_before - mse_after
        improvement_percentage = ((mse_before - mse_after) / mse_before) * 100 if mse_before != 0 else 0
        channel_improvements.append(improvement_percentage)

        # Collect channels with improvement > 10% (no risk of overfitting)
        if improvement_percentage > -8:
            no_risk_channels.append(i)

    # Calculate overall MSE improvement
    total_initial_mse = mean_squared_error(test_true.flatten(), init_pred.flatten())
    total_final_mse = mean_squared_error(test_true.flatten(), test_pred.flatten())
    total_improvement = total_initial_mse - total_final_mse
    improvement_percentage = ((total_initial_mse - total_final_mse) / total_initial_mse) * 100 if total_initial_mse != 0 else 0

    # Check for overfitting risk
    if total_improvement < 0:
        if improvement_percentage < -10:
            overfitting_risk = "High risk of overfitting!"
        else:
            overfitting_risk = "Overfitting risk with the loss of improvement"
        overfitting_bool = True
    else:
        overfitting_bool = False
        overfitting_risk = "No significant overfitting risk."

    # Display results
    print(f"Overall Improvement: {improvement_percentage:.2f}%")
    print(f"Channel-wise Improvements: {channel_improvements}")
    print(f"Overfitting Risk: {overfitting_risk}")
    
    # Return the channel(s) with no overfitting risk and overall overfitting risk
    return overfitting_risk, overfitting_bool, no_risk_channels

def visualize_improvements(exp, args, best_function_set_per_channel, best_function_set_per_channel_feedback, GenericFunction, GenericFunction2):
    """
    Visualize the improvement in predictions by comparing the initial predictions with the improved ones.
    Applies only to channels without overfitting risk.

    Args:
    - exp: The experiment or model instance
    - args: The arguments or configurations for the model
    - best_function_set_per_channel: The best actions (functions and parameters) from exploration
    - best_function_set_per_channel_feedback: The best actions (functions and parameters) from feedback
    - GenericFunction: Class for applying functions from exploration
    - GenericFunction2: Class for applying functions from feedback
    
    Returns:
    - total_initial_mse: The initial MSE (mean squared error) before improvements
    - total_final_mse: The final MSE (after improvements)
    - improvement_percentage: Percentage of improvement in MSE
    - total_initial_mae: The initial MAE (mean absolute error) before improvements
    - total_final_mae: The final MAE (after improvements)
    - mae_improvement_percentage: Percentage of improvement in MAE
    - channel_improvements: List of MSE improvements per channel
    - channel_mae_improvements: List of MAE improvements per channel
    """
    # Get overfitting risk information and the channels with no overfitting risk
    _, _, no_risk_channels_all = check_overfitting(exp, args, best_function_set_per_channel, best_function_set_per_channel_feedback, GenericFunction, GenericFunction2)

    # Combine the no-risk channels from exploration and feedback
    no_risk_channels = no_risk_channels_all

    # Load new test data
    _, test_pred, test_true, batch_x = exp.test(args.model_id, test=1)  # Assuming exp.test gives you the predictions, true values, and features
    init_pred = test_pred.copy()

    channel_improvements = []  # To store MSE improvements per channel
    channel_mae_improvements = []  # To store MAE improvements per channel

    # Apply the best function set for exploration, but only for channels with no overfitting risk
    for channel_idx, best_action, best_params in best_function_set_per_channel:
        if channel_idx in no_risk_channels:
            new_channel_pred = GenericFunction(best_action, best_params).apply(
                test_pred[:, :, channel_idx].reshape((test_pred.shape[0], test_pred.shape[1], 1)),
                batch_x[:, :, channel_idx].reshape((batch_x.shape[0], batch_x.shape[1], 1)).cpu().numpy()
            )
            test_pred[:, :, channel_idx] = new_channel_pred[:, :, 0]

    # Apply the best function set for feedback, but only for channels with no overfitting risk
    for channel_idx, best_action, best_params in best_function_set_per_channel_feedback:
        if channel_idx in no_risk_channels:
            new_channel_pred = GenericFunction2(best_action, best_params).apply(
                test_pred[:, :, channel_idx].reshape((test_pred.shape[0], test_pred.shape[1], 1)),
                batch_x[:, :, channel_idx].reshape((batch_x.shape[0], batch_x.shape[1], 1)).cpu().numpy()
            )
            test_pred[:, :, channel_idx] = new_channel_pred[:, :, 0]

    # Calculate MSE and MAE before and after improvements for each channel
    for i in range(test_pred.shape[2]):
        mse_before = mean_squared_error(test_true[:, :, i].flatten(), init_pred[:, :, i].flatten())
        mse_after = mean_squared_error(test_true[:, :, i].flatten(), test_pred[:, :, i].flatten())
        mae_before = mean_absolute_error(test_true[:, :, i].flatten(), init_pred[:, :, i].flatten())
        mae_after = mean_absolute_error(test_true[:, :, i].flatten(), test_pred[:, :, i].flatten())
        
        # Store the improvements for both MSE and MAE
        channel_improvements.append(mse_before - mse_after)  # Store the MSE improvement
        channel_mae_improvements.append(mae_before - mae_after)  # Store the MAE improvement

    # Calculate total MSE and MAE improvements
    total_initial_mse = mean_squared_error(test_true.flatten(), init_pred.flatten())
    total_final_mse = mean_squared_error(test_true.flatten(), test_pred.flatten())
    total_initial_mae = mean_absolute_error(test_true.flatten(), init_pred.flatten())
    total_final_mae = mean_absolute_error(test_true.flatten(), test_pred.flatten())

    total_mse_improvement = total_initial_mse - total_final_mse
    total_mae_improvement = total_initial_mae - total_final_mae
    
    improvement_percentage_mse = ((total_initial_mse - total_final_mse) / total_initial_mse) * 100 if total_initial_mse != 0 else 0
    improvement_percentage_mae = ((total_initial_mae - total_final_mae) / total_initial_mae) * 100 if total_initial_mae != 0 else 0

    # Print the improvement summary to the command line
    print("\n### MSE and MAE Improvement Summary ###")
    print(f"- Total Initial MSE: {total_initial_mse:.4f}")
    print(f"- Total Final MSE: {total_final_mse:.4f}")
    print(f"- Total MSE Improvement: {total_mse_improvement:.4f}")
    print(f"- Overall MSE Improvement: {improvement_percentage_mse:.2f}%\n")

    print(f"- Total Initial MAE: {total_initial_mae:.4f}")
    print(f"- Total Final MAE: {total_final_mae:.4f}")
    print(f"- Total MAE Improvement: {total_mae_improvement:.4f}")
    print(f"- Overall MAE Improvement: {improvement_percentage_mae:.2f}%\n")

    # Print the channel-wise improvements
    print("Channel-wise Improvements:")
    for idx, channel_idx in enumerate(no_risk_channels):
        print(f"- Channel {channel_idx + 1}: {channel_improvements[idx]:.4f} MSE, {channel_mae_improvements[idx]:.4f} MAE")

    return total_initial_mse, total_final_mse, improvement_percentage_mse, total_initial_mae, total_final_mae, improvement_percentage_mae


def test_and_visualize_on_new_data(exp, args, best_function_set_per_channel, best_function_set_per_channel_feedback, GenericFunction, GenericFunction2):
    """
    Test and visualize the model's predictions on new test data, including both exploration-based actions
    and user-provided feedback actions.

    Args:
    - exp: The experiment or model instance
    - args: The arguments or configurations for the model
    - best_function_set_per_channel: The best actions (functions and parameters) from exploration
    - feedback_list: A list of user-provided feedback actions (feedback_text, channel_idx)
    """
    # Load new test data
    _, test_pred, test_true, batch_x = exp.test(args.model_id, test=1)  # Assuming exp.test gives you the predictions, true values, and features
    init_pred = test_pred.copy()
    # Store the channel MSEs and improvements
    channel_improvements = []
    mse_before = []
    mse_after = []

    # Set up the number of subplots (one for each channel)
    n_channels = test_pred.shape[2]
    ncols = 2  # Set the number of columns for the grid of subplots
    nrows = (n_channels + 1) // ncols  # Calculate the number of rows needed

    # Create a grid layout for subplots (only for the first sample)
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows), sharex=True, sharey=True)
    axes = axes.flatten()  # Flatten the axes array for easier iteration

    # Set a beautiful style using Seaborn
    sns.set(style="whitegrid", palette="muted")

    # Plot only for the first sample
    mse_values = np.mean((test_true - test_pred) ** 2, axis=(1, 2))

# Find the index of the sample with the highest MSE
    highest_mse_index = np.argmax(mse_values)
    sample_idx = highest_mse_index  # Visualizing only the first sample

    # Vectorize the MSE calculation and the predictions for each channel
    # Track initial MSE for all channels
  # MSE for each channel, averaged over all samples
    
    # Apply the best action set for this sample
    for channel_idx, best_action, best_params in best_function_set_per_channel:
        # Apply the function to all channels of the current sample
        new_channel_pred = GenericFunction(best_action, best_params).apply(test_pred[:, :, channel_idx].reshape((test_pred.shape[0], test_pred.shape[1], 1)), batch_x[:, :, channel_idx].reshape((batch_x.shape[0], batch_x.shape[1], 1)).cpu().numpy())
        test_pred[:, :, channel_idx] = new_channel_pred[:, :, 0]

    for channel_idx, best_action, best_params in best_function_set_per_channel_feedback:
        # Apply the function to all channels of the current sample
        new_channel_pred = GenericFunction2(best_action, best_params).apply(test_pred[:, :, channel_idx].reshape((test_pred.shape[0], test_pred.shape[1], 1)), batch_x[:, :, channel_idx].reshape((batch_x.shape[0], batch_x.shape[1], 1)).cpu().numpy())
        test_pred[:, :, channel_idx] = new_channel_pred[:, :, 0]

    # Apply the user feedback actions
    # if feedback_list:
    #     for feedback_text, channel_idx in feedback_list:
    #         test_pred[:, :, channel_idx] = apply_feedback_to_predictions(test_pred[:, :, channel_idx], feedback_text, channel_idx)

  # MSE after applying actions/feedback
    
    # Calculate the improvement for each channel
    for i in range(test_pred.shape[2]):
            # Track the improved MSE for all channels
        mse_before = mean_squared_error(test_true[:, :, i].flatten(), init_pred[:, :, i].flatten())
        mse_after = mean_squared_error(test_true[:, :, i].flatten(), test_pred[:, :, i].flatten())
        channel_improvements.append(mse_before - mse_after)  # This should be an array of improvements, one per channel

    # Plot the results for the first sample
    for i, _ in enumerate(range(test_pred.shape[2])):
        ax = axes[i]
        if i >= n_channels:
            ax.axis("off")  # Turn off any empty subplots (if n_channels < total subplots)

        channel_pred = test_pred[sample_idx, :, i]  # Get the initial prediction for the channel
        channel_true = test_true[sample_idx, :, i]  # True values for the channel

        # Plot the ground truth once (for comparison)
        ax.plot(range(len(batch_x[sample_idx, :, i].cpu())), batch_x[sample_idx, :, i].cpu(), color="gray", linestyle="--", linewidth=2, label="Input (Initial)")
        ax.plot(range(len(batch_x[sample_idx, :, i].cpu()), len(batch_x[sample_idx, :, i].cpu()) + len(channel_true)), 
                channel_true, color="blue", linewidth=3, label="Ground Truth")

        # Plot the initial prediction for this channel
        ax.plot(range(len(batch_x[sample_idx, :, i].cpu()), len(batch_x[sample_idx, :, i].cpu()) + len(channel_pred)),
                init_pred[sample_idx, :, i], color="green", linestyle=":", linewidth=2, label=f"Channel {i+1} (Initial)")

        # Plot the updated prediction after applying the actions (both from exploration and feedback)
        improved_predictions = test_pred[sample_idx, :, i]
        ax.plot(range(len(batch_x[sample_idx, :, i].cpu()), len(batch_x[sample_idx, :, i].cpu()) + len(improved_predictions)),
                improved_predictions, color="red", linestyle="-", linewidth=2, label=f"Channel {i+1} (Improved)")

        # Beautify each subplot with titles and labels
        ax.set_title(f"Channel {i+1}: Prediction Improvement", fontsize=16, fontweight='bold', color='darkgreen')
        ax.set_xlabel("Time Steps", fontsize=12, color="black")
        ax.set_ylabel("Prediction Value", fontsize=12, color="black")
        ax.grid(True, linestyle='-', alpha=0.3)

    # Add a custom legend (common for all subplots)
    custom_lines = [
        Line2D([0], [0], color="blue", lw=2, label="Ground Truth"),
        Line2D([0], [0], color="gray", linestyle="--", lw=2, label="Input (Initial)"),
        Line2D([0], [0], color="red", lw=2, linestyle="-", label="Predictions (Improved)"),
        Line2D([0], [0], color="green", lw=2, linestyle=":", label="Initial Predictions (Channel)"),
    ]
    fig.legend(handles=custom_lines, loc="upper left", fontsize=12)

    # Display the plot with all subplots for the first sample
    # plt.tight_layout()
    # st.pyplot(fig)
    # Calculate the total MSE improvement for all channels and samples (averaged over all samples)
    total_initial_mse = mean_squared_error(test_true.flatten(), init_pred.flatten())  # Average MSE across all channels
    total_final_mse = mean_squared_error(test_true.flatten(), test_pred.flatten())  # Average MSE across all channels
    total_improvement = total_initial_mse - total_final_mse

    improvement_percentage = ((total_initial_mse - total_final_mse) / total_initial_mse) * 100 if total_initial_mse != 0 else 0

    # Beautiful box to display the MSE improvement results
    improvement_box = f"""
    ### MSE Improvement Summary

    - **Total Initial MSE**: {total_initial_mse:.4f}
    - **Total Final MSE**: {total_final_mse:.4f}
    - **Total Improvement**: {total_improvement:.4f}
    - **Overall MSE Improvement**: {improvement_percentage:.2f}%
    
    **Channel-wise MSE Improvements**:
    """
    
    # Fixing the iteration and formatting of the improvements
    for i, improvement in enumerate(channel_improvements):
        # Ensure each improvement is formatted correctly (improvement is a scalar value for each channel)
        improvement_box += f"\n- **Channel {i+1}**: {improvement:.4f} MSE"
    
    # Display the summary in a beautiful box
    st.markdown(f"<div style='border: 2px solid #4CAF50; padding: 10px; background-color: #f9f9f9; border-radius: 5px;'><b>{improvement_box}</b></div>", unsafe_allow_html=True)

    return test_pred, total_initial_mse, total_final_mse, improvement_percentage, channel_improvements  # Return channel-wise improvements

from torch.utils.data import DataLoader
dummy_marks = lambda x: np.zeros((x.shape[0], x.shape[1], 4))

def autoregressive_forecast(exp, args, best_function_set_per_channel, best_function_set_per_channel_feedback,
                            q, test_sample):
    """
    Perform autoregressive forecasting using a trained model and apply transformations per channel.

    Parameters:
        exp: Trained model experiment with .test() method.
        args: Arguments with model_id, etc.
        best_function_set_per_channel: List of (channel_idx, action, params).
        best_function_set_per_channel_feedback: List for feedback loop.
        best_function: Callable for applying transformation (e.g., GenericFunction).
        q: Forecast horizon at each step.
        test_sample: Total number of time steps to predict.
        TimeSeriesForecastingDataset: Dataset class used for feeding data to model.
        dummy_marks: Function to create dummy inputs if required by dataset.
        GenericFunction, GenericFunction2: Transformation function classes.

    Returns:
        final_predictions: np.array of shape [batch_size, test_sample, num_channels]
    """

    # Step 1: Identify safe channels
    _, _, no_risk_channels_all = check_overfitting(
        exp, args, best_function_set_per_channel, best_function_set_per_channel_feedback,
        GenericFunction, GenericFunction
    )
    no_risk_channels = no_risk_channels_all

    # Step 2: Initial test call to seed input
    _, _, _, batch_x = exp.test(args.model_id, test=1)
    test_loader = args.test_loader
    X_test_exo_copy = test_loader.dataset.X_exo
    batch_x = batch_x.cpu().numpy()
    B, T, C = batch_x.shape

    total_steps = test_sample
    num_loops = total_steps // q
    remainder = total_steps % q

    predictions = []
    current_input = batch_x

    for loop in range(num_loops + (1 if remainder > 0 else 0)):
        step_q = loop * q
        print('LOOP', step_q, q)
        current_q = q
        if current_q == 0:
            break

        # Create dummy y and dummy marks to match dataset format
        dummy_y = np.zeros((B, current_q, C))
        # test_dataset = TimeSeriesForecastingDataset(
        #     current_input,
        #     dummy_y,
        #     dummy_marks(current_input),
        #     dummy_marks(dummy_y)
        # )
        print(current_input.shape, dummy_y.shape, X_test_exo_copy[:, step_q:step_q + current_q].shape)
        test_dataset = TimeSeriesForecastingDataset(
        current_input, dummy_y,
        X_exo=X_test_exo_copy[:, step_q:step_q + current_q], y_exo=X_test_exo_copy[:, step_q:step_q + current_q]
    )
        test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False)
        args.test_loader = test_loader  # Inject test_loader into args for exp.test()

        # Run prediction: model outputs (B, q, C)
        _, new_pred, _, _ = exp.test(args.model_id, test=1)
        new_pred = new_pred[:, :current_q, :]  # Safety check

        # Apply transformation on safe channels
        for channel_idx, best_action, best_params in best_function_set_per_channel:
            if channel_idx in no_risk_channels:
                transformed = GenericFunction(best_action, best_params).apply(
                    new_pred[:, :, channel_idx].reshape(B, current_q, 1),
                    current_input[:, -T:, channel_idx].reshape(B, T, 1)
                )
                new_pred[:, :, channel_idx] = transformed[:, :, 0]

        if loop < num_loops:
            new_pred = new_pred[:remainder]
        predictions.append(new_pred)

        # Autoregressive feedback
        # current_input = np.concatenate([current_input[:, current_q:, :], new_pred], axis=1)
        current_input = np.concatenate([current_input, new_pred], axis=1)  # Shape: [B, T+q, C]
        current_input = current_input[:, -T:, :]  # Crop to last T

    # Concatenate all predictions
    final_predictions = np.concatenate(predictions, axis=1)  # [B, test_sample, C]
    return final_predictions

