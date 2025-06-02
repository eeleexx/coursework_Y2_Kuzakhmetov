import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import os

def create_sector_comparisons(stock_data: pd.DataFrame, news_data: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Create sector-specific comparison visualizations
    """
    # Create sector mappings
    sectors = {
        'Technology': ['AMD', 'INTC'],
        'Banking': ['JPM', 'BAC'],
        'Healthcare': ['PFE', 'JNJ']
    }
    
    # Prepare data for sector analysis
    sector_stats = []
    for sector, companies in sectors.items():
        for company in companies:
            if company in news_data:
                df = news_data[company]
                stats = {
                    'Sector': sector,
                    'Company': company,
                    'Articles_Per_Day': len(df) / len(df['date'].unique()),
                    'Avg_Headline_Length': df['title'].str.len().mean(),
                    'Avg_Summary_Length': df['summary'].str.len().mean()
                }
                sector_stats.append(stats)
    
    sector_df = pd.DataFrame(sector_stats)
    
    # Create sector comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Articles per day by sector
    plt.subplot(2, 2, 1)
    sns.barplot(data=sector_df, x='Sector', y='Articles_Per_Day', hue='Company')
    plt.title('News Coverage by Sector')
    plt.xticks(rotation=45)
    
    # Plot 2: Content length comparison
    plt.subplot(2, 2, 2)
    sector_df_melted = pd.melt(sector_df, 
                              id_vars=['Sector', 'Company'],
                              value_vars=['Avg_Headline_Length', 'Avg_Summary_Length'],
                              var_name='Content Type',
                              value_name='Length')
    sns.boxplot(data=sector_df_melted, x='Sector', y='Length', hue='Content Type')
    plt.title('Content Length by Sector')
    plt.xticks(rotation=45)
    
    # Plot 3: Sentiment volatility by sector
    sentiment_volatility = []
    for sector, companies in sectors.items():
        for company in companies:
            if company in news_data:
                df = news_data[company]
                df['date'] = pd.to_datetime(df['date'])
                daily_sentiment = df.groupby('date')['title'].count().reset_index()
                volatility = daily_sentiment['title'].std()
                sentiment_volatility.append({
                    'Sector': sector,
                    'Company': company,
                    'Volatility': volatility
                })
    
    volatility_df = pd.DataFrame(sentiment_volatility)
    plt.subplot(2, 2, 3)
    sns.barplot(data=volatility_df, x='Sector', y='Volatility', hue='Company')
    plt.title('News Volume Volatility by Sector')
    plt.xticks(rotation=45)
    
    # Plot 4: Stock price correlation heatmap
    correlations = []
    for sector, companies in sectors.items():
        for company1 in companies:
            stock1 = stock_data[stock_data['symbol'] == company1]['Close'].pct_change()
            for company2 in companies:
                if company1 != company2:
                    stock2 = stock_data[stock_data['symbol'] == company2]['Close'].pct_change()
                    corr = stock1.corr(stock2)
                    correlations.append({
                        'Sector': sector,
                        'Pair': f'{company1}-{company2}',
                        'Correlation': corr
                    })
    
    corr_df = pd.DataFrame(correlations)
    plt.subplot(2, 2, 4)
    sns.barplot(data=corr_df, x='Sector', y='Correlation')
    plt.title('Intra-Sector Stock Price Correlation')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sector_comparison.png'))
    plt.close()
    
    # Create sentiment trend plots
    plt.figure(figsize=(15, 15))
    for i, (sector, companies) in enumerate(sectors.items(), 1):
        plt.subplot(3, 1, i)
        for company in companies:
            if company in news_data:
                df = news_data[company]
                df['date'] = pd.to_datetime(df['date'])
                daily_sentiment = df.groupby('date')['title'].count().rolling(30).mean()
                plt.plot(daily_sentiment.index, daily_sentiment.values, label=company)
        plt.title(f'{sector} Sector - 30-Day Moving Average News Volume')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sector_trends.png'))
    plt.close() 