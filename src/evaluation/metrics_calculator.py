import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

def generate_classification_report(y_true, y_pred, model_name: str):
    """
    Prints a detailed classification report for a given model.
    """
    print(f"--- Classification Report for {model_name} ---")
    report = classification_report(y_true, y_pred, target_names=['Ham', 'Spam'])
    print(report)

def plot_confusion_matrix(y_true, y_pred, model_name: str):
    """
    Generates and displays a confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()