import pandas as pd
import numpy as np
from typing import Dict, List
import os
from models.knn_model import StockKNN
from features.feature_engineering import prepare_features_for_prediction
from analysis.sentiment_analyzer import SentimentAnalyzer
from sklearn.model_selection import train_test_split
import json


def run_knn_analysis(
    stock_data: pd.DataFrame,
    news_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    knn_dir: str,
    prediction_type: str = 'classification',
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
    # Check if results already exist
    results_file = os.path.join(knn_dir, 'knn_results.json')
    if os.path.exists(results_file) and not force_rerun:
        print("\nLoading existing KNN results...")
        with open(results_file, 'r') as f:
            return json.load(f)

    results = {}
    sentiment_analyzer = SentimentAnalyzer()

    for symbol in symbols:
        print(f"\nAnalyzing {symbol} with KNN...")

        # Check if results already exist for this symbol
        model_file = os.path.join(
            knn_dir, 'models', f'{symbol}_knn_model.joblib')
        pred_plot = os.path.join(
            knn_dir, 'plots', f'{symbol}_knn_predictions.png')
        if os.path.exists(model_file) and os.path.exists(pred_plot) and not force_rerun:
            print(f"Results already exist for {symbol}, skipping...")
            continue

        # Get symbol specific data
        symbol_stock = stock_data[stock_data['symbol'] == symbol].copy()
        symbol_news = news_data.get(symbol)

        if symbol_news is None or len(symbol_stock) == 0:
            print(f"Skipping {symbol} - insufficient data")
            continue

        # Calculate sentiment scores
        print(
            f"Calculating sentiment scores for {len(symbol_news)} news items...")
        symbol_news['headline_sentiment'] = symbol_news['title'].apply(
            sentiment_analyzer.analyze_text)
        symbol_news['summary_sentiment'] = symbol_news['summary'].apply(
            sentiment_analyzer.analyze_text)

        # Calculate daily sentiment
        print("Aggregating daily sentiment...")
        daily_sentiment = symbol_news.groupby('date').agg({
            'headline_sentiment': 'mean',
            'summary_sentiment': 'mean',
            'title': 'count'
        }).reset_index()
        daily_sentiment = daily_sentiment.rename(
            columns={'title': 'news_count'})

        # Prepare features
        print(
            f"Raw data points - Stock: {len(symbol_stock)}, News: {len(daily_sentiment)}")
        try:
            X, y = prepare_features_for_prediction(
                symbol_stock, daily_sentiment)
            print(f"Valid data points after feature preparation: {len(X)}")
        except Exception as e:
            print(f"Error during feature preparation: {str(e)}")
            print("Skipping symbol due to feature preparation error")
            continue

        # Check if we have enough data
        if len(X) < 5:  # Lower requirement from 10 to 5
            print(
                f"Skipping {symbol} - not enough valid data points after feature preparation")
            print("This can happen due to:")
            print("- Missing data at the start for calculating moving averages")
            print("- Days without news coverage")
            print("- Missing or invalid values in the data")
            continue

        # Split data
        # Ensure at least 10 training samples
        test_size = min(0.2, 1.0 - 10.0/len(X))
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        try:
            # Initialize and train KNN model
            knn = StockKNN(n_neighbors=min(5, len(X_train)//2),  # Adjust n_neighbors based on data size
                           prediction_type=prediction_type)
            train_metrics = knn.train(X_train, y_train)

            # Make predictions on test set
            y_pred = knn.predict(X_test)
            test_metrics = knn.evaluate(y_test, y_pred)

            # Ensure directories exist
            os.makedirs(os.path.join(knn_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(knn_dir, 'plots'), exist_ok=True)

            # Save model
            knn.save_model(os.path.join(knn_dir, 'models'), symbol)

            # Plot feature importance
            feature_importance = knn.plot_feature_importance(
                X_test, y_test,
                os.path.join(knn_dir, 'plots'),
                symbol
            )

            # Plot predictions
            knn.plot_predictions(
                X_test, y_test,
                os.path.join(knn_dir, 'plots'),
                symbol
            )

            # Store results
            results[symbol] = {
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'feature_importance': feature_importance,
                'data_points': len(X),
                'training_size': len(X_train),
                'test_size': len(X_test)
            }

            # Print results
            print(f"\nResults for {symbol}:")
            print("\nTraining Metrics:")
            for metric, value in train_metrics.items():
                print(f"{metric}: {value:.4f}")

            print("\nTest Metrics:")
            for metric, value in test_metrics.items():
                print(f"{metric}: {value:.4f}")

            print("\nTop 5 Important Features:")
            importance_series = pd.Series(feature_importance)
            for feature, importance in importance_series.nlargest(5).items():
                print(f"{feature}: {importance:.4f}")

        except Exception as e:
            print(
                f"Error during model training/evaluation for {symbol}: {str(e)}")
            print("Skipping to next symbol...")
            continue

    # Save results to JSON
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results
