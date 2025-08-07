import numpy as np
import pandas as pd
import io

def concatenate_predictions(pred_test, test_indices_list, exp, args, target_cols, channel_id):
    """
    Combine predictions for one or multiple target columns into a DataFrame.
    
    Parameters:
    - pred_test: list of np.arrays, each predictions for a group,
                 shape (n_samples, n_targets) or (n_samples,) or (a,b,c) for 3D
    - test_indices_list: list of indices corresponding to predictions
    - exp, args: for validation metrics (unchanged)
    - target_cols: list of target column names (if single target, can be str or list of one str)
    
    Returns:
    - stacked_preds: np.array with shape (total_samples, n_targets)
    - y_tr_pred, y_tr, y_val_clean, y_val_pred_clean: validation outputs
    """
    
    # Normalize target_cols to list if needed
    if isinstance(target_cols, str):
        target_cols = [target_cols]

    pred_rows = []  # Will store tuples (index, predictions_array)
    
    for preds, indices in zip(pred_test, test_indices_list):
        preds = np.array(preds)  # Ensure numpy array
        
        if preds.ndim == 1:
            # Single target, shape (n_samples,) -> (n_samples, 1)
            preds = preds[:, None]
        elif preds.ndim == 3:
            # Flatten first two dims: (a, b, c) -> (a*b, c)
            a, b, c = preds.shape
            preds = preds.reshape(a * b, c)
        elif preds.ndim > 3:
            raise ValueError(f"Predictions have unexpected shape {preds.shape}, expected max 3D array.")
        # else preds.ndim == 2 is OK
        # print('final pred', preds.shape)
        n_samples, n_targets = preds.shape
        
        # Check that n_targets matches number of target columns
        # if n_targets != len(target_cols):
        #     raise ValueError(f"Number of prediction targets ({n_targets}) does not match "
        #                      f"number of target columns ({len(target_cols)})")
        
        # If indices are None or all None, generate default indices
        if indices is None or all(idx is None for idx in indices):
            indices = list(range(n_samples))
        
        for i, idx in enumerate(indices):
            pred_rows.append((idx, preds[i]))
    
    # Sort predictions by index
    pred_rows = sorted(pred_rows, key=lambda x: x[0])
    
    # Extract sorted indices and stacked predictions
    sorted_indices = [idx for idx, _ in pred_rows]
    stacked_preds = np.vstack([p for _, p in pred_rows])
    
    # Build submission DataFrame with columns for each target
    submission_dict = {"id": sorted_indices}
    for i, col in enumerate(target_cols):
        submission_dict[col] = stacked_preds[:, i]
    submission = pd.DataFrame(submission_dict).sort_values("id")
    
    # Save to CSV buffer (optional)
    buffer = io.StringIO()
    submission.to_csv(buffer, index=False)
    buffer.seek(0)
    
    # Validation metrics (unchanged)
    _, y_val_pred, y_val, _ = exp.validation(args.model_id, test=1)
    # _, y_tr_pred, y_tr, _ = exp.validation(args.model_id, test=1)
    # print(y_val_pred.shape, y_val.shape)
    y_val = y_val[:, :, channel_id]
    y_val_pred = y_val_pred[:, :, channel_id]
    mask = ~np.isnan(y_val)
    y_val_clean = y_val[mask]
    y_val_pred_clean = y_val_pred[mask]
    # print(y_val_clean.shape, y_val_pred_clean.shape)
    y_tr_pred = y_val_pred
    y_tr = y_val
    
    return stacked_preds, y_tr_pred, y_tr, y_val_clean, y_val_pred_clean
