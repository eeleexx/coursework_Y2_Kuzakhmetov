import pandas as pd
import numpy as np
from typing import Dict, List
import os
from models.lstm_model import StockLSTM
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


def run_lstm_analysis(
    stock_data: pd.DataFrame,
    news_data: Dict[str, pd.DataFrame],
    symbols: List[str],
    lstm_dir: str,
    vader_sentiment_dir: str,
    prediction_type: str = 'regression',
    force_rerun: bool = False
) -> Dict:
    """
    Run LSTM analysis for each symbol with different feature combinations

    Args:
        stock_data: DataFrame with stock prices
        news_data: Dictionary of news DataFrames
        symbols: List of stock symbols to analyze
        lstm_dir: Directory for LSTM results
        vader_sentiment_dir: Directory containing precomputed sentiment values
        prediction_type: Either 'regression' or 'classification'
        force_rerun: If True, rerun analysis even if results exist
    """
    results = {}

    for symbol in symbols:
        print(f"\nAnalyzing {symbol} with LSTM...")

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
                variation_dir = os.path.join(lstm_dir, variation)
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
                    # Create variation directories
                    model_dir = os.path.join(variation_dir, 'models')
                    plots_dir = os.path.join(variation_dir, 'plots')
                    os.makedirs(model_dir, exist_ok=True)
                    os.makedirs(plots_dir, exist_ok=True)

                    # Initialize and train model
                    lstm = StockLSTM(data, filters=features)
                    history = lstm.fit(
                        lookback=10,
                        train_size=0.8,
                        epochs=300,
                        model_dir=model_dir,
                        ticker=symbol
                    )

                    # Save model explicitly
                    model_path = os.path.join(
                        model_dir, f'{symbol}_lstm_model.keras')
                    lstm.model.save(model_path)

                    # Generate and save plots
                    metrics = lstm.plot_multiple_graphs(
                        ticker=symbol, output_dir=plots_dir)

                    # Store results
                    if symbol not in results:
                        results[symbol] = {}

                    results[symbol][variation] = {
                        'metrics': {
                            'train_rmse': metrics['train_rmse'],
                            'train_mae': metrics['train_mae'],
                            'test_rmse': metrics['test_rmse'],
                            'test_mae': metrics['test_mae']
                        },
                        'history': {
                            'loss': history.history['loss'],
                            'val_loss': history.history['val_loss']
                        },
                        'data_points': len(data),
                        'test_predictions': metrics['test_predictions'],
                        'test_actual': metrics['test_actual']
                    }

                    # Save individual results
                    results_file = os.path.join(
                        variation_dir, f'{symbol}_lstm_metrics.json')
                    with open(results_file, 'w') as f:
                        json.dump(results[symbol][variation], f, indent=4)

                    print(f"\nResults for {symbol} - {variation}:")
                    print(f"Training RMSE: {metrics['train_rmse']:.4f}")
                    print(f"Training MAE: {metrics['train_mae']:.4f}")
                    print(f"Test RMSE: {metrics['test_rmse']:.4f}")
                    print(f"Test MAE: {metrics['test_mae']:.4f}")

                except Exception as e:
                    print(
                        f"Error during {variation} analysis for {symbol}: {str(e)}")
                    continue

        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue

    # Save all results
    results_file = os.path.join(lstm_dir, 'lstm_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)

    return results
