import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import Dict, List
from .sentiment_analyzer import SentimentAnalyzer


def categorize_sentiment(score: float) -> str:
    """Categorize sentiment scores into descriptive categories"""
    if score <= -0.5:
        return "Very Negative"
    elif score <= -0.1:
        return "Negative"
    elif score <= 0.1:
        return "Neutral"
    elif score <= 0.5:
        return "Positive"
    else:
        return "Very Positive"


def analyze_sentiment_details(news_data: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Analyze sentiment details for each company and save to CSV files
    """
    sentiment_analyzer = SentimentAnalyzer()

    for symbol, df in news_data.items():
        # Create detailed sentiment analysis
        detailed_sentiment = df.copy()

        # Calculate sentiments
        detailed_sentiment['headline_sentiment'] = detailed_sentiment['title'].apply(
            sentiment_analyzer.analyze_text)
        detailed_sentiment['summary_sentiment'] = detailed_sentiment['summary'].apply(
            sentiment_analyzer.analyze_text)

        # Add sentiment categories
        detailed_sentiment['headline_category'] = detailed_sentiment['headline_sentiment'].apply(
            categorize_sentiment)
        detailed_sentiment['summary_category'] = detailed_sentiment['summary_sentiment'].apply(
            categorize_sentiment)

        # Sort by date and sentiment
        detailed_sentiment = detailed_sentiment.sort_values(
            ['date', 'headline_sentiment'], ascending=[True, False])

        # Add a column for sentiment difference (headline vs summary)
        detailed_sentiment['sentiment_difference'] = detailed_sentiment['headline_sentiment'] - \
            detailed_sentiment['summary_sentiment']
        detailed_sentiment['clickbait_score'] = abs(
            detailed_sentiment['sentiment_difference'])

        # Mark potential clickbait (high difference between headline and summary sentiment)
        detailed_sentiment['is_potential_clickbait'] = detailed_sentiment['clickbait_score'] > 0.5

        # Select and reorder columns
        columns = [
            'date', 'title', 'headline_sentiment', 'headline_category',
            'summary', 'summary_sentiment', 'summary_category',
            'sentiment_difference', 'clickbait_score', 'is_potential_clickbait'
        ]

        # Save to CSV
        output_file = os.path.join(
            output_dir, f'{symbol}_sentiment_details.csv')
        detailed_sentiment[columns].to_csv(output_file, index=False)

        # Generate summary statistics
        summary_stats = {
            'Total Articles': len(detailed_sentiment),
            'Average Headline Sentiment': detailed_sentiment['headline_sentiment'].mean(),
            'Average Summary Sentiment': detailed_sentiment['summary_sentiment'].mean(),
            'Potential Clickbait Articles': detailed_sentiment['is_potential_clickbait'].sum(),
            'Most Positive Headlines': detailed_sentiment.nlargest(5, 'headline_sentiment')[['date', 'title', 'headline_sentiment']],
            'Most Negative Headlines': detailed_sentiment.nsmallest(5, 'headline_sentiment')[['date', 'title', 'headline_sentiment']],
            'Sentiment Distribution': detailed_sentiment['headline_category'].value_counts()
        }

        # Save summary statistics
        summary_file = os.path.join(
            output_dir, f'{symbol}_sentiment_summary.txt')
        with open(summary_file, 'w') as f:
            f.write(f"Sentiment Analysis Summary for {symbol}\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Total Articles: {summary_stats['Total Articles']}\n")
            f.write(
                f"Average Headline Sentiment: {summary_stats['Average Headline Sentiment']:.3f}\n")
            f.write(
                f"Average Summary Sentiment: {summary_stats['Average Summary Sentiment']:.3f}\n")
            f.write(
                f"Potential Clickbait Articles: {summary_stats['Potential Clickbait Articles']}\n\n")

            f.write("Sentiment Distribution:\n")
            f.write(str(summary_stats['Sentiment Distribution']) + "\n\n")

            f.write("Most Positive Headlines:\n")
            f.write(str(summary_stats['Most Positive Headlines']) + "\n\n")

            f.write("Most Negative Headlines:\n")
            f.write(str(summary_stats['Most Negative Headlines']) + "\n")
