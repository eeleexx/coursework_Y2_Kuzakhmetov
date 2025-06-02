import json
from typing import Dict, List, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import numpy as np
import pandas as pd
from analysis.knn_analysis import run_knn_analysis
from analysis.sector_visualization import create_sector_comparisons
from analysis.competitor_analysis import analyze_cpu_competition
from analysis.sentiment_details import analyze_sentiment_details
from analysis.sentiment_analyzer import SentimentAnalyzer
from models.knn_model import StockKNN
from models.lstm_model import StockLSTM
import os
import sys
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')


sys.stderr = stderr

SYMBOLS = ['AMD', 'BAC', 'INTC', 'JNJ', 'JPM', 'PFE']
DATA_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'results')

VADER_DIR = os.path.join(RESULTS_DIR, 'vader_analysis')
VADER_PLOTS = os.path.join(VADER_DIR, 'plots')
VADER_SENTIMENT = os.path.join(VADER_DIR, 'sentiment')
VADER_COMPETITION = os.path.join(VADER_DIR, 'competition')
VADER_SECTORS = os.path.join(VADER_DIR, 'sectors')
VADER_PREDICTIONS = os.path.join(VADER_DIR, 'predictions')

KNN_DIR = os.path.join(RESULTS_DIR, 'knn_analysis')

# New KNN variation directories
KNN_PRICE_ONLY = os.path.join(KNN_DIR, 'price_only')
KNN_PRICE_HEADLINE = os.path.join(KNN_DIR, 'price_headline')
KNN_PRICE_SUMMARY = os.path.join(KNN_DIR, 'price_summary')
KNN_PRICE_HEADLINE_SUMMARY = os.path.join(KNN_DIR, 'price_headline_summary')
KNN_PRICE_HEADLINE_NO_CLICKBAIT = os.path.join(
    KNN_DIR, 'price_headline_no_clickbait')
KNN_PRICE_SUMMARY_NO_CLICKBAIT = os.path.join(
    KNN_DIR, 'price_summary_no_clickbait')
KNN_PRICE_HEADLINE_SUMMARY_NO_CLICKBAIT = os.path.join(
    KNN_DIR, 'price_headline_summary_no_clickbait')

LSTM_DIR = os.path.join(RESULTS_DIR, 'lstm_analysis')

# New LSTM variation directories
LSTM_PRICE_ONLY = os.path.join(LSTM_DIR, 'price_only')
LSTM_PRICE_HEADLINE = os.path.join(LSTM_DIR, 'price_headline')
LSTM_PRICE_SUMMARY = os.path.join(LSTM_DIR, 'price_summary')
LSTM_PRICE_HEADLINE_SUMMARY = os.path.join(LSTM_DIR, 'price_headline_summary')
LSTM_PRICE_HEADLINE_NO_CLICKBAIT = os.path.join(
    LSTM_DIR, 'price_headline_no_clickbait')
LSTM_PRICE_SUMMARY_NO_CLICKBAIT = os.path.join(
    LSTM_DIR, 'price_summary_no_clickbait')
LSTM_PRICE_HEADLINE_SUMMARY_NO_CLICKBAIT = os.path.join(
    LSTM_DIR, 'price_headline_summary_no_clickbait')

os.makedirs(DATA_DIR, exist_ok=True)

os.makedirs(VADER_DIR, exist_ok=True)
os.makedirs(VADER_PLOTS, exist_ok=True)
os.makedirs(VADER_SENTIMENT, exist_ok=True)
os.makedirs(VADER_COMPETITION, exist_ok=True)
os.makedirs(VADER_SECTORS, exist_ok=True)
os.makedirs(VADER_PREDICTIONS, exist_ok=True)

# Create KNN directories
os.makedirs(KNN_DIR, exist_ok=True)
os.makedirs(KNN_PRICE_ONLY, exist_ok=True)
os.makedirs(KNN_PRICE_HEADLINE, exist_ok=True)
os.makedirs(KNN_PRICE_SUMMARY, exist_ok=True)
os.makedirs(KNN_PRICE_HEADLINE_SUMMARY, exist_ok=True)
os.makedirs(KNN_PRICE_HEADLINE_NO_CLICKBAIT, exist_ok=True)
os.makedirs(KNN_PRICE_SUMMARY_NO_CLICKBAIT, exist_ok=True)
os.makedirs(KNN_PRICE_HEADLINE_SUMMARY_NO_CLICKBAIT, exist_ok=True)

# Create LSTM directories
os.makedirs(LSTM_DIR, exist_ok=True)
os.makedirs(LSTM_PRICE_ONLY, exist_ok=True)
os.makedirs(LSTM_PRICE_HEADLINE, exist_ok=True)
os.makedirs(LSTM_PRICE_SUMMARY, exist_ok=True)
os.makedirs(LSTM_PRICE_HEADLINE_SUMMARY, exist_ok=True)
os.makedirs(LSTM_PRICE_HEADLINE_NO_CLICKBAIT, exist_ok=True)
os.makedirs(LSTM_PRICE_SUMMARY_NO_CLICKBAIT, exist_ok=True)
os.makedirs(LSTM_PRICE_HEADLINE_SUMMARY_NO_CLICKBAIT, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load stock and news data from CSV files
    """
    print("Loading data...")

    stock_data = pd.read_csv(os.path.join(DATA_DIR, 'stock_data.csv'))
    stock_data['Date'] = pd.to_datetime(
        stock_data['Date'], utc=True).dt.strftime('%Y-%m-%d')

    news_data = {}
    for symbol in SYMBOLS:
        news_file = os.path.join(DATA_DIR, f'{symbol}_news.csv')
        if os.path.exists(news_file):
            df = pd.read_csv(news_file)
            df['date'] = pd.to_datetime(
                df['date'], utc=True).dt.strftime('%Y-%m-%d')
            news_data[symbol] = df

    return stock_data, news_data


def analyze_sentiment_impact(stock_data: pd.DataFrame, news_data: Dict[str, pd.DataFrame]) -> None:
    """
    Analyze the impact of news sentiment on stock prices
    """
    sentiment_analyzer = SentimentAnalyzer()

    for symbol in SYMBOLS:
        if symbol not in news_data:
            print(f"\nSkipping {symbol} - no news data available")
            continue

        print(f"\nAnalyzing {symbol}...")

        symbol_stock = stock_data[stock_data['symbol'] == symbol].copy()
        if len(symbol_stock) == 0:
            print(f"No stock data available for {symbol}")
            continue

        symbol_news = news_data[symbol].copy()

        symbol_news['headline_sentiment'] = symbol_news['title'].apply(
            sentiment_analyzer.analyze_text)
        symbol_news['summary_sentiment'] = symbol_news['summary'].apply(
            sentiment_analyzer.analyze_text)

        daily_sentiment = symbol_news.groupby('date').agg({
            'headline_sentiment': 'mean',
            'summary_sentiment': 'mean',
            'title': 'count'
        }).reset_index()
        daily_sentiment = daily_sentiment.rename(
            columns={'title': 'news_count'})

        merged_data = pd.merge(symbol_stock, daily_sentiment,
                               left_on='Date', right_on='date', how='left')

        merged_data = merged_data.assign(
            headline_sentiment=merged_data['headline_sentiment'].fillna(0),
            summary_sentiment=merged_data['summary_sentiment'].fillna(0),
            news_count=merged_data['news_count'].fillna(0)
        )

        merged_data['return'] = merged_data['Close'].pct_change()

        merged_data = merged_data.dropna(subset=['return'])

        if len(merged_data) < 2:
            print(f"Not enough data points for {symbol} after merging")
            continue

        headline_corr, headline_pvalue = stats.pearsonr(
            merged_data['headline_sentiment'], merged_data['return'])
        summary_corr, summary_pvalue = stats.pearsonr(
            merged_data['summary_sentiment'], merged_data['return'])

        headline_mse = np.mean(
            (merged_data['return'] - merged_data['headline_sentiment'])**2)
        headline_rmse = np.sqrt(headline_mse)
        headline_mae = np.mean(
            np.abs(merged_data['return'] - merged_data['headline_sentiment']))

        summary_mse = np.mean(
            (merged_data['return'] - merged_data['summary_sentiment'])**2)
        summary_rmse = np.sqrt(summary_mse)
        summary_mae = np.mean(
            np.abs(merged_data['return'] - merged_data['summary_sentiment']))

        ks_headline = stats.ks_2samp(
            merged_data['return'], merged_data['headline_sentiment'])
        ks_summary = stats.ks_2samp(
            merged_data['return'], merged_data['summary_sentiment'])

        print(f"\nResults for {symbol}:")
        print("\nHeadline Sentiment Analysis:")
        print(
            f"Correlation with returns: {headline_corr:.4f} (p-value: {headline_pvalue:.4f})")
        print(f"MSE: {headline_mse:.6f}")
        print(f"RMSE: {headline_rmse:.6f}")
        print(f"MAE: {headline_mae:.6f}")
        print(
            f"KS test statistic: {ks_headline.statistic:.4f} (p-value: {ks_headline.pvalue:.4f})")

        print("\nSummary Sentiment Analysis:")
        print(
            f"Correlation with returns: {summary_corr:.4f} (p-value: {summary_pvalue:.4f})")
        print(f"MSE: {summary_mse:.6f}")
        print(f"RMSE: {summary_rmse:.6f}")
        print(f"MAE: {summary_mae:.6f}")
        print(
            f"KS test statistic: {ks_summary.statistic:.4f} (p-value: {ks_summary.pvalue:.4f})")

        print("\nNews Coverage Statistics:")
        print(f"Total number of news articles: {len(symbol_news)}")
        print(f"Days with news coverage: {len(daily_sentiment)}")
        print(
            f"Average news articles per day with news: {daily_sentiment['news_count'].mean():.2f}")

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.plot(merged_data['Date'], merged_data['Close'],
                 label='Stock Price', color='blue')
        plt.plot(merged_data['Date'], merged_data['headline_sentiment'] * merged_data['Close'].mean(),
                 label='Headline Sentiment (scaled)', color='red', alpha=0.5)
        plt.title(f'{symbol} Stock Price and Sentiment Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price / Sentiment')
        plt.legend()
        plt.xticks(rotation=45)

        plt.subplot(2, 1, 2)
        sns.histplot(data=merged_data, x='headline_sentiment',
                     label='Headlines', alpha=0.5, color='red')
        sns.histplot(data=merged_data, x='summary_sentiment',
                     label='Summaries', alpha=0.5, color='blue')
        plt.title(f'{symbol} Sentiment Distribution')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(
            VADER_PLOTS, f'{symbol}_sentiment_analysis.png'))
        plt.close()

        plt.figure(figsize=(15, 10))

        plt.subplot(2, 1, 1)
        plt.scatter(merged_data['headline_sentiment'], merged_data['return'],
                    alpha=0.5, label='Headlines')
        plt.title(f'{symbol} Returns vs Sentiment')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Daily Return')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.bar(merged_data['Date'], merged_data['news_count'])
        plt.title(f'{symbol} Daily News Coverage')
        plt.xlabel('Date')
        plt.ylabel('Number of News Articles')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(
            VADER_PLOTS, f'{symbol}_additional_analysis.png'))
        plt.close()


def plot_vader_predictions(stock_data: pd.DataFrame, news_data: Dict[str, pd.DataFrame], output_dir: str):
    """
    Plot naive VADER sentiment-based predictions vs actual stock prices for each symbol.
    Naive prediction: use scaled headline sentiment as the predicted price movement.
    """
    for symbol in SYMBOLS:
        if symbol not in news_data:
            continue
        symbol_stock = stock_data[stock_data['symbol'] == symbol].copy()
        if len(symbol_stock) == 0:
            continue
        symbol_news = news_data[symbol].copy()

        symbol_stock['Date'] = pd.to_datetime(symbol_stock['Date'])

        symbol_news['headline_sentiment'] = symbol_news['title'].apply(
            SentimentAnalyzer().analyze_text)
        daily_sentiment = symbol_news.groupby('date').agg(
            {'headline_sentiment': 'mean'}).reset_index()

        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])

        merged = pd.merge(symbol_stock, daily_sentiment,
                          left_on='Date', right_on='date', how='left')
        merged = merged.assign(
            headline_sentiment=merged['headline_sentiment'].fillna(0))

        min_price, max_price = merged['Close'].min(), merged['Close'].max()
        pred = (merged['headline_sentiment'] - merged['headline_sentiment'].min()) / (
            merged['headline_sentiment'].max() - merged['headline_sentiment'].min() + 1e-8)
        pred = pred * (max_price - min_price) + min_price

        plt.figure(figsize=(15, 6))
        plt.plot(merged['Date'], merged['Close'], label='Actual', color='blue')
        plt.plot(merged['Date'], pred,
                 label='Predicted (VADER naive)', color='orange')
        plt.title(f'VADER Naive Prediction vs Actual [{symbol}]')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'{symbol}_vader_predictions.png'))
        plt.close()


def run_lstm_analysis(stock_data: pd.DataFrame, news_data: Dict[str, pd.DataFrame], symbols: List[str], lstm_dir: str, vader_sentiment_dir: str, prediction_type: str, force_rerun: bool):
    """
    Run LSTM analysis for each symbol using both price and sentiment data
    """
    plots_dir = os.path.join(lstm_dir, 'plots')
    models_dir = os.path.join(lstm_dir, 'models')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    results = {}

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

    for symbol in symbols:
        if symbol not in news_data:
            continue

        print(f"\nRunning LSTM analysis for {symbol}...")

        symbol_stock = stock_data[stock_data['symbol'] == symbol].copy()
        if len(symbol_stock) == 0:
            print(f"No stock data available for {symbol}")
            continue

        try:
            # Load precomputed sentiment
            sentiment_file = os.path.join(
                vader_sentiment_dir, f'{symbol}_sentiment_details.csv')
            if not os.path.exists(sentiment_file):
                print(f"Sentiment file not found for {symbol}")
                continue

            sentiment_data = pd.read_csv(sentiment_file)
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])

            # Convert dates to datetime
            symbol_stock['Date'] = pd.to_datetime(symbol_stock['Date'])

            # Aggregate daily sentiment
            daily_sentiment = sentiment_data.groupby('date').agg({
                'headline_sentiment': 'mean',
                'summary_sentiment': 'mean',
                'title': 'count',
                'is_potential_clickbait': 'max'
            }).reset_index()
            daily_sentiment = daily_sentiment.rename(
                columns={'title': 'news_count'})

            # Merge with stock data
            merged_data = pd.merge(symbol_stock, daily_sentiment,
                                   left_on='Date', right_on='date', how='left')
            merged_data = merged_data.fillna(0)

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
                        'metrics': metrics,
                        'history': {
                            'loss': history.history['loss'],
                            'val_loss': history.history['val_loss']
                        },
                        'data_points': len(data)
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


def main():
    stock_data, news_data = load_data()

    print("\nRunning VADER sentiment analysis...")
    analyze_sentiment_impact(stock_data, news_data)
    analyze_sentiment_details(news_data, VADER_SENTIMENT)
    analyze_cpu_competition(stock_data, news_data, VADER_COMPETITION)
    create_sector_comparisons(stock_data, news_data, VADER_SECTORS)
    plot_vader_predictions(stock_data, news_data,
                           os.path.join(os.path.dirname(__file__), '..', 'results', 'vader_analysis', 'predictions'))

    print("\nRunning KNN analysis...")
    run_knn_analysis(
        stock_data=stock_data,
        news_data=news_data,
        symbols=SYMBOLS,
        knn_dir=KNN_DIR,
        vader_sentiment_dir=VADER_SENTIMENT,
        prediction_type='regression',
        force_rerun=False
    )

    print("\nRunning LSTM analysis...")
    run_lstm_analysis(
        stock_data=stock_data,
        news_data=news_data,
        symbols=SYMBOLS,
        lstm_dir=LSTM_DIR,
        vader_sentiment_dir=VADER_SENTIMENT,
        prediction_type='regression',
        force_rerun=False
    )


if __name__ == "__main__":
    main()
