import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Union, Tuple


class SentimentAnalyzer:
    def __init__(self):
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')

        self.sia = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> float:
        """Analyze sentiment of a single text and return compound score"""
        if pd.isna(text):
            return 0.0

        scores = self.sia.polarity_scores(text)
        return scores['compound']

    def analyze_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze sentiment for both headlines and summaries
        Returns DataFrame with sentiment scores for each
        """
        result_df = df.copy()

        result_df['headline_sentiment'] = result_df['title'].apply(
            self.analyze_text)

        result_df['summary_sentiment'] = result_df['summary'].apply(
            self.analyze_text)

        result_df['sentiment_diff'] = result_df['headline_sentiment'] - \
            result_df['summary_sentiment']

        return result_df

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate statistical metrics including MSE, RMSE, MAE"""
        mse = np.mean((y_true - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_true - y_pred))

        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(y_true, y_pred)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'KS_statistic': ks_statistic,
            'p_value': p_value
        }

    def analyze_clickbait_effect(self, sentiment_df: pd.DataFrame, returns_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Analyze the potential clickbait effect by comparing headline vs summary sentiment impact on returns
        """
        results = {}

        merged_df = pd.merge(sentiment_df, returns_df, on=[
                             'date', 'symbol'], how='inner')

        for symbol in merged_df['symbol'].unique():
            symbol_data = merged_df[merged_df['symbol'] == symbol]

            sentiment_diff = symbol_data['headline_sentiment'] - \
                symbol_data['summary_sentiment']

            median_diff = sentiment_diff.median()
            high_diff = symbol_data[sentiment_diff > median_diff]
            low_diff = symbol_data[sentiment_diff <= median_diff]

            metrics = {
                'headline_vs_returns': self.calculate_metrics(
                    symbol_data['headline_sentiment'].values,
                    symbol_data['return'].values
                ),
                'summary_vs_returns': self.calculate_metrics(
                    symbol_data['summary_sentiment'].values,
                    symbol_data['return'].values
                ),
                'high_vs_low_diff': self.calculate_metrics(
                    high_diff['return'].values,
                    low_diff['return'].values
                )
            }

            # Additional statistics
            metrics['avg_headline_impact'] = np.corrcoef(
                symbol_data['headline_sentiment'],
                symbol_data['return']
            )[0, 1]

            metrics['avg_summary_impact'] = np.corrcoef(
                symbol_data['summary_sentiment'],
                symbol_data['return']
            )[0, 1]

            metrics['clickbait_score'] = np.mean(np.abs(sentiment_diff))

            results[symbol] = metrics

        return results

    def get_sentiment_summary(self, sentiment_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Generate comprehensive summary statistics for sentiment analysis
        """
        summary = {}

        for sentiment_type in ['headline_sentiment', 'summary_sentiment']:
            data = sentiment_df[sentiment_type]

            summary[sentiment_type] = {
                'mean': data.mean(),
                'std': data.std(),
                'median': data.median(),
                'positive_ratio': (data > 0).mean(),
                'negative_ratio': (data < 0).mean(),
                'neutral_ratio': (data == 0).mean(),
                # Strong positive
                'extreme_positive_ratio': (data > 0.5).mean(),
                # Strong negative
                'extreme_negative_ratio': (data < -0.5).mean(),
            }

            quartiles = data.quantile([0.25, 0.5, 0.75])
            summary[sentiment_type].update({
                'q1': quartiles[0.25],
                'q2': quartiles[0.50],  # median
                'q3': quartiles[0.75],
                'iqr': quartiles[0.75] - quartiles[0.25]
            })

        sentiment_diff = sentiment_df['headline_sentiment'] - \
            sentiment_df['summary_sentiment']
        summary['sentiment_difference'] = {
            'mean': sentiment_diff.mean(),
            'std': sentiment_diff.std(),
            'median': sentiment_diff.median(),
            # Headline more positive
            'positive_diff_ratio': (sentiment_diff > 0).mean(),
            # Summary more positive
            'negative_diff_ratio': (sentiment_diff < 0).mean(),
            # Significant differences
            'significant_diff_ratio': (np.abs(sentiment_diff) > 0.2).mean()
        }

        return summary
