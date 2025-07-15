"""
@author: Kashaf Gulzar

classifier file imported in acoustic_classification.py
"""

import numpy as np
import pandas as pd


def compute_metrics(confusion_matrix):
    """Calculates specificity, sensitivity, accuracy and uar from confusion matrix
    Confusion matrix of form [[tn, fp]
                              [fn, tp]]
    args:
      confusion_matrix: 2 by 2 nd-array
      output: tuple of float (specificity, sensitivity, accuracy, uar)
    """
    cm = confusion_matrix
    tp, tn = cm[1, 1], cm[0, 0]
    fn, fp = cm[1, 0], cm[0, 1]

    sensitivity = (tp / (tp + fn))*100
    specificity = (tn / (fp + tn))*100
    accuracy = ((tp + tn) / (tp + tn + fp + fn))*100
    uar = (specificity + sensitivity) / 2.0

    metrics_dict = dict(accuracy=accuracy, uar=uar,
                        sensitivity=sensitivity, specificity=specificity)
    return metrics_dict


# Calculate mean and standard deviation for each metric
def calculate_metrics_summary(metrics_dict, random_seeds: list):
    summary = {
        metric: {
            'mean': np.mean([metrics_dict[seed][metric] for seed in random_seeds]),
            'std_dev': np.std([metrics_dict[seed][metric] for seed in random_seeds])
        }
        for metric in metrics_dict[random_seeds[0]]
    }
    summary_df = pd.DataFrame(summary)

    summary_combined = summary_df
    # Transpose the dataframe to make 'mean' and 'std' rows into columns
    summary_combined = summary_combined.T
    # Concatenate 'mean' and 'std' values and format them as 'mean ± std'
    summary_combined['mean ± std'] = round(summary_combined['mean'], 2).astype(str) + ' ± ' + round(summary_combined['std_dev'],2).astype(str)
    # Drop the 'mean' and 'std' rows
    summary_combined = summary_combined.drop(columns=['mean', 'std_dev'])
    # Transpose the dataframe back to the original format
    summary_combined = summary_combined.T

    return summary_df, summary_combined