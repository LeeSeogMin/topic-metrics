# common_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_metrics(metrics_df, metrics):
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='Model', y=metric, data=metrics_df)
        plt.title(f'Model {metric} Comparison')
        plt.show()

        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Num_Topics', y=metric, hue='Model', data=metrics_df)
        plt.title(f'{metric} vs Number of Topics')
        plt.show()

def analyze_correlations(metrics_df, metrics):
    correlation_matrix = metrics_df[metrics].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation between Evaluation Metrics')
    plt.show()