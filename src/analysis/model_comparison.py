from src.models.knn_model import KNN
from src.models.lstm_model import LSTM
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def run_model_comparison():
    # Define model variations
    variations = [
        'price_only',                    # Only historical price data
        'price_headline',                # Price + headline sentiment
        'price_summary',                 # Price + summary sentiment
        'price_headline_summary',        # Price + both sentiments
        # Price + headline (excluding clickbait)
        'price_headline_no_clickbait',
        # Price + summary (excluding clickbait)
        'price_summary_no_clickbait',
        # Price + both (excluding clickbait)
        'price_headline_summary_no_clickbait'
    ]

    results = []

    for ticker in ['AMD', 'INTC', 'JPM', 'BAC', 'PFE', 'JNJ']:
        for variation in variations:
            # Load and prepare data based on variation
            if variation == 'price_only':
                features = ['Close']  # Only price data
            elif variation == 'price_headline':
                features = ['Close', 'headline_sentiment']
            elif variation == 'price_summary':
                features = ['Close', 'summary_sentiment']
            elif variation == 'price_headline_summary':
                features = ['Close', 'headline_sentiment', 'summary_sentiment']
            elif variation == 'price_headline_no_clickbait':
                features = ['Close', 'headline_sentiment']
                # Filter out clickbait headlines
                # Add your clickbait filtering logic here
            elif variation == 'price_summary_no_clickbait':
                features = ['Close', 'summary_sentiment']
                # Filter out clickbait summaries
                # Add your clickbait filtering logic here
            elif variation == 'price_headline_summary_no_clickbait':
                features = ['Close', 'headline_sentiment', 'summary_sentiment']
                # Filter out clickbait
                # Add your clickbait filtering logic here

            # Run LSTM model
            lstm_model = LSTM(filters=features)
            lstm_metrics = lstm_model.train_and_evaluate(ticker)

            # Run KNN model
            knn_model = KNN(filters=features)
            knn_metrics = knn_model.train_and_evaluate(ticker)

            # Store results
            results.append({
                'Ticker': ticker,
                'Variation': variation,
                'LSTM_RMSE': lstm_metrics['RMSE'],
                'LSTM_MAE': lstm_metrics['MAE'],
                'LSTM_R2': lstm_metrics['R2'],
                'KNN_RMSE': knn_metrics['RMSE'],
                'KNN_MAE': knn_metrics['MAE'],
                'KNN_R2': knn_metrics['R2']
            })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    os.makedirs('results/model_comparison', exist_ok=True)
    results_df.to_csv(
        'results/model_comparison/model_variations_results.csv', index=False)

    # Generate LaTeX table
    generate_latex_table(results_df)


def generate_latex_table(results_df):
    # Calculate average metrics for each variation
    avg_metrics = results_df.groupby('Variation').mean()

    # Generate LaTeX table
    latex_table = """
\\begin{table}[H]
\\centering
\\caption{Model Performance Comparison Across Different Feature Combinations}
\\begin{tabular}{lcccccc}
\\toprule
\\multirow{2}{*}{Features} & \\multicolumn{3}{c}{LSTM} & \\multicolumn{3}{c}{KNN} \\\\
\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}
 & RMSE & MAE & R2 & RMSE & MAE & R2 \\\\
\\midrule
"""

    for variation in avg_metrics.index:
        metrics = avg_metrics.loc[variation]
        latex_table += f"{variation.replace('_', ' ').title()} & "
        latex_table += f"{metrics['LSTM_RMSE']:.4f} & {metrics['LSTM_MAE']:.4f} & {metrics['LSTM_R2']:.4f} & "
        latex_table += f"{metrics['KNN_RMSE']:.4f} & {metrics['KNN_MAE']:.4f} & {metrics['KNN_R2']:.4f} \\\\\n"

    latex_table += """\\bottomrule
\\end{tabular}
\\end{table}"""

    # Save LaTeX table
    with open('results/model_comparison/model_comparison_table.tex', 'w') as f:
        f.write(latex_table)


if __name__ == "__main__":
    run_model_comparison()
