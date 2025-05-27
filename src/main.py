from models.lstm_model import StockLSTM
from models.knn_model import StockKNN
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.sentiment_details import analyze_sentiment_details
from analysis.competitor_analysis import analyze_cpu_competition
from analysis.sector_visualization import create_sector_comparisons
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# --- Configuration ---
SYMBOLS = ['AMD', 'BAC', 'INTC', 'JNJ', 'JPM', 'PFE']
DATA_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), '..', 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
SENTIMENT_DIR = os.path.join(RESULTS_DIR, 'sentiment')
COMPETITION_DIR = os.path.join(RESULTS_DIR, 'competition')
SECTOR_DIR = os.path.join(RESULTS_DIR, 'sectors')

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SENTIMENT_DIR, exist_ok=True)
os.makedirs(COMPETITION_DIR, exist_ok=True)
os.makedirs(SECTOR_DIR, exist_ok=True)


def load_data() -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Load stock and news data from CSV files
    """
    print("Loading data...")

    # Load stock data
    stock_data = pd.read_csv(os.path.join(DATA_DIR, 'stock_data.csv'))
    stock_data['Date'] = pd.to_datetime(
        stock_data['Date'], utc=True).dt.strftime('%Y-%m-%d')

    # Load news data for each symbol
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

        # Get stock data for the symbol
        symbol_stock = stock_data[stock_data['symbol'] == symbol].copy()
        if len(symbol_stock) == 0:
            print(f"No stock data available for {symbol}")
            continue

        symbol_news = news_data[symbol].copy()

        # Analyze sentiment of headlines and summaries
        symbol_news['headline_sentiment'] = symbol_news['title'].apply(
            sentiment_analyzer.analyze_text)
        symbol_news['summary_sentiment'] = symbol_news['summary'].apply(
            sentiment_analyzer.analyze_text)

        # Calculate daily average sentiment
        daily_sentiment = symbol_news.groupby('date').agg({
            'headline_sentiment': 'mean',
            'summary_sentiment': 'mean',
            'title': 'count'
        }).reset_index()
        daily_sentiment = daily_sentiment.rename(
            columns={'title': 'news_count'})

        # Merge with stock data
        merged_data = pd.merge(symbol_stock, daily_sentiment,
                               left_on='Date', right_on='date', how='left')

        # Fill missing values
        merged_data = merged_data.assign(
            headline_sentiment=merged_data['headline_sentiment'].fillna(0),
            summary_sentiment=merged_data['summary_sentiment'].fillna(0),
            news_count=merged_data['news_count'].fillna(0)
        )

        # Calculate daily returns
        merged_data['return'] = merged_data['Close'].pct_change()

        # Remove first row (no return)
        merged_data = merged_data.dropna(subset=['return'])

        if len(merged_data) < 2:
            print(f"Not enough data points for {symbol} after merging")
            continue

        # Perform statistical tests
        headline_corr, headline_pvalue = stats.pearsonr(
            merged_data['headline_sentiment'], merged_data['return'])
        summary_corr, summary_pvalue = stats.pearsonr(
            merged_data['summary_sentiment'], merged_data['return'])

        # Calculate MSE, RMSE, MAE for headline sentiment vs returns
        headline_mse = np.mean(
            (merged_data['return'] - merged_data['headline_sentiment'])**2)
        headline_rmse = np.sqrt(headline_mse)
        headline_mae = np.mean(
            np.abs(merged_data['return'] - merged_data['headline_sentiment']))

        # Calculate MSE, RMSE, MAE for summary sentiment vs returns
        summary_mse = np.mean(
            (merged_data['return'] - merged_data['summary_sentiment'])**2)
        summary_rmse = np.sqrt(summary_mse)
        summary_mae = np.mean(
            np.abs(merged_data['return'] - merged_data['summary_sentiment']))

        # Perform Kolmogorov-Smirnov test
        ks_headline = stats.ks_2samp(
            merged_data['return'], merged_data['headline_sentiment'])
        ks_summary = stats.ks_2samp(
            merged_data['return'], merged_data['summary_sentiment'])

        # Print results
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

        # Additional statistics
        print("\nNews Coverage Statistics:")
        print(f"Total number of news articles: {len(symbol_news)}")
        print(f"Days with news coverage: {len(daily_sentiment)}")
        print(
            f"Average news articles per day with news: {daily_sentiment['news_count'].mean():.2f}")

        # Create plots
        plt.figure(figsize=(15, 10))

        # Plot 1: Stock Price and Sentiment Over Time
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

        # Plot 2: Sentiment Distribution
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
            PLOTS_DIR, f'{symbol}_sentiment_analysis.png'))
        plt.close()

        # Create additional plots
        plt.figure(figsize=(15, 10))

        # Plot 3: Returns vs Sentiment Scatter Plot
        plt.subplot(2, 1, 1)
        plt.scatter(merged_data['headline_sentiment'], merged_data['return'],
                    alpha=0.5, label='Headlines')
        plt.title(f'{symbol} Returns vs Sentiment')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Daily Return')
        plt.legend()

        # Plot 4: News Count Over Time
        plt.subplot(2, 1, 2)
        plt.bar(merged_data['Date'], merged_data['news_count'])
        plt.title(f'{symbol} Daily News Coverage')
        plt.xlabel('Date')
        plt.ylabel('Number of News Articles')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(os.path.join(
            PLOTS_DIR, f'{symbol}_additional_analysis.png'))
        plt.close()


def main():
    # Load data
    stock_data, news_data = load_data()

    # Analyze sentiment impact
    analyze_sentiment_impact(stock_data, news_data)

    # Generate detailed sentiment analysis
    analyze_sentiment_details(news_data, SENTIMENT_DIR)

    # Analyze AMD vs Intel competition
    analyze_cpu_competition(stock_data, news_data, COMPETITION_DIR)

    # Create sector comparisons
    create_sector_comparisons(stock_data, news_data, SECTOR_DIR)


if __name__ == "__main__":
    main()
