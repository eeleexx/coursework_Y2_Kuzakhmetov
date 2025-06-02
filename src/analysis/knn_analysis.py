import pandas as pd
import numpy as np
from typing import Dict, List
import os
from models.knn_model import StockKNN
from features.feature_engineering import prepare_features_for_prediction
import json


def load_precomputed_sentiment(symbol: str, vader_sentiment_dir: str) -> pd.DataFrame:
    """
    Load precomputed sentiment values from CSV file
    """
    sentiment_file = os.path.join(
        vader_sentiment_dir, f'{symbol}_sentiment_details.csv')
    if not os.path.exists(sentiment_file):
        raise FileNotFoundError(f"Sentiment file not found for {symbol}")

    df = pd.read_csv(sentiment_file)
    df['date'] = pd.to_datetime(df['date'])
    return df


def run_knn_analysis(
    stock_data: pd.DataFrame,
    news_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    knn_dir: str,
    vader_sentiment_dir: str,
    prediction_type: str = 'regression',
    force_rerun: bool = False
) -> Dict:
    """
    Run KNN analysis for each symbol with different feature combinations

    Args:
        stock_data: DataFrame with stock prices
        news_data: Dictionary of news DataFrames
        symbols: List of stock symbols to analyze
        knn_dir: Directory for KNN results
        vader_sentiment_dir: Directory containing precomputed sentiment values
        prediction_type: Either 'regression' or 'classification'
        force_rerun: If True, rerun analysis even if results exist
    """
    results = {}

    for symbol in symbols:
        print(f"\nAnalyzing {symbol} with KNN...")

        symbol_stock = stock_data[stock_data['symbol'] == symbol].copy()
        if len(symbol_stock) == 0:
            print(f"Skipping {symbol} - insufficient stock data")
            continue

        try:
            # Load precomputed sentiment
            sentiment_data = load_precomputed_sentiment(
                symbol, vader_sentiment_dir)

            # Convert dates to datetime
            symbol_stock['Date'] = pd.to_datetime(symbol_stock['Date'])

            # Aggregate daily sentiment
            daily_sentiment = sentiment_data.groupby('date').agg({
                'headline_sentiment': 'mean',
                'summary_sentiment': 'mean',
                'title': 'count',
                'is_potential_clickbait': 'sum'
            }).reset_index()
            daily_sentiment = daily_sentiment.rename(
                columns={'title': 'news_count'})

            # Merge with stock data
            merged_data = pd.merge(symbol_stock, daily_sentiment,
                                   left_on='Date', right_on='date', how='left')
            merged_data = merged_data.fillna(0)

            # Define feature combinations
            feature_combinations = {
                'price_only': ['Close'],
                'price_headline': ['Close', 'headline_sentiment'],
                'price_summary': ['Close', 'summary_sentiment'],
                'price_headline_summary': ['Close', 'headline_sentiment', 'summary_sentiment'],
                'price_headline_no_clickbait': ['Close', 'headline_sentiment'],
                'price_summary_no_clickbait': ['Close', 'summary_sentiment'],
                'price_headline_summary_no_clickbait': ['Close', 'headline_sentiment', 'summary_sentiment']
            }

            # Run analysis for each feature combination
            for variation, features in feature_combinations.items():
                print(f"\nRunning {variation} variation...")

                # Create variation directory
                variation_dir = os.path.join(knn_dir, variation)
                os.makedirs(variation_dir, exist_ok=True)

                # Prepare data based on variation
                if 'no_clickbait' in variation:
                    # Filter out days with clickbait
                    filtered_data = merged_data[merged_data['is_potential_clickbait'] == 0].copy(
                    )
                else:
                    filtered_data = merged_data.copy()

                data = filtered_data[features].copy()

                try:
                    knn = StockKNN(n_neighbors=5)
                    train_metrics = knn.fit(data, lookback=10, train_size=0.8)
                    test_metrics = knn.evaluate()

                    # Save model and plots
                    model_dir = os.path.join(variation_dir, 'models')
                    plots_dir = os.path.join(variation_dir, 'plots')
                    os.makedirs(model_dir, exist_ok=True)
                    os.makedirs(plots_dir, exist_ok=True)

                    knn.save_model(model_dir, symbol)
                    metrics = knn.plot_multiple_graphs(plots_dir, symbol)

                    # Store results
                    if symbol not in results:
                        results[symbol] = {}

                    results[symbol][variation] = {
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics,
                        'data_points': len(data),
                        'plot_metrics': metrics
                    }

                    print(f"\nResults for {symbol} - {variation}:")
                    print("\nTraining Metrics:")
                    for metric, value in train_metrics.items():
                        print(f"{metric}: {value:.4f}")

                    print("\nTest Metrics:")
                    for metric, value in test_metrics.items():
                        print(f"{metric}: {value:.4f}")

                except Exception as e:
                    print(
                        f"Error during {variation} analysis for {symbol}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Save all results
    results_file = os.path.join(knn_dir, 'knn_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results
