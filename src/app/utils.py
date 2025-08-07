import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def plot_true_vs_pred(y_true, y_pred, title="True vs Predicted"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.4)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(title)
    st.pyplot(fig)

def compute_metrics(y_true, y_pred):
    return {
        "R²": r2_score(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }