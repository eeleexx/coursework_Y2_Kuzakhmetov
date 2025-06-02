import pandas as pd
import numpy as np
from typing import Dict, List
import os
from models.knn_model import StockKNN
from features.feature_engineering import prepare_features_for_prediction
from analysis.sentiment_analyzer import SentimentAnalyzer
from sklearn.model_selection import train_test_split
import json

sentiment_analyzer = SentimentAnalyzer()


def run_knn_analysis(
    stock_data: pd.DataFrame,
    news_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    knn_dir: str,
    prediction_type: str = 'regression',
    force_rerun: bool = False
) -> Dict:
    """
    Run KNN analysis for each symbol

    Args:
        stock_data: DataFrame with stock prices
        news_data: Dictionary of news DataFrames
        symbols: List of stock symbols to analyze
        knn_dir: Directory for KNN results
        prediction_type: Either 'regression' or 'classification'
        force_rerun: If True, rerun analysis even if results exist
    """

    results_file = os.path.join(knn_dir, 'knn_results.json')
    if os.path.exists(results_file) and not force_rerun:
        print("\nLoading existing KNN results...")
        with open(results_file, 'r') as f:
            return json.load(f)

    results = {}

    for symbol in symbols:
        print(f"\nAnalyzing {symbol} with KNN...")

        model_file = os.path.join(
            knn_dir, 'models', f'{symbol}_knn_model.joblib')
        pred_plot = os.path.join(
            knn_dir, 'plots', f'{symbol}_knn_predictions.png')
        if os.path.exists(model_file) and os.path.exists(pred_plot) and not force_rerun:
            print(f"Results already exist for {symbol}, skipping...")
            continue

        symbol_stock = stock_data[stock_data['symbol'] == symbol].copy()
        symbol_news = news_data.get(symbol)

        if symbol_news is None or len(symbol_stock) == 0:
            print(f"Skipping {symbol} - insufficient data")
            continue

        symbol_stock['Date'] = pd.to_datetime(symbol_stock['Date'])
        symbol_news['date'] = pd.to_datetime(symbol_news['date'])

        print(
            f"Calculating sentiment scores for {len(symbol_news)} news items...")
        symbol_news['headline_sentiment'] = symbol_news['title'].apply(
            sentiment_analyzer.analyze_text)
        symbol_news['summary_sentiment'] = symbol_news['summary'].apply(
            sentiment_analyzer.analyze_text)

        print("Aggregating daily sentiment...")
        daily_sentiment = symbol_news.groupby('date').agg({
            'headline_sentiment': 'mean',
            'summary_sentiment': 'mean',
            'title': 'count'
        }).reset_index()
        daily_sentiment = daily_sentiment.rename(
            columns={'title': 'news_count'})

        merged_data = pd.merge(symbol_stock, daily_sentiment,
                               left_on='Date', right_on='date', how='left')
        merged_data = merged_data.fillna(0)

        features = ['Close', 'headline_sentiment']
        data = merged_data[features].copy()

        try:
            knn = StockKNN(n_neighbors=5)
            train_metrics = knn.fit(data, lookback=10, train_size=0.8)

            test_metrics = knn.evaluate()

            os.makedirs(os.path.join(knn_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(knn_dir, 'plots'), exist_ok=True)

            knn.save_model(os.path.join(knn_dir, 'models'), symbol)

            metrics = knn.plot_multiple_graphs(
                os.path.join(knn_dir, 'plots'),
                symbol
            )

            results[symbol] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'data_points': len(data),
                'plot_metrics': metrics
            }

            print(f"\nResults for {symbol}:")
            print("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                print(f"{metric}: {value:.4f}")

            print("\nTest Metrics:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value:.4f}")

        except Exception as e:
            print(
                f"Error during model training/evaluation for {symbol}: {str(e)}")
            print("Skipping to next symbol...")
            continue

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results
