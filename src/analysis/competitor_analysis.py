import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple
import os


def analyze_cpu_competition(stock_data: pd.DataFrame, news_data: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
    Analyze the competition between AMD and Intel, focusing on CPU-related news
    """

    amd_stock = stock_data[stock_data['symbol'] == 'AMD'].copy()
    intel_stock = stock_data[stock_data['symbol'] == 'INTC'].copy()

    amd_stock['Date'] = pd.to_datetime(amd_stock['Date'])
    intel_stock['Date'] = pd.to_datetime(intel_stock['Date'])

    amd_stock['relative_performance'] = (
        amd_stock['Close'] / amd_stock['Close'].iloc[0]) * 100
    intel_stock['relative_performance'] = (
        intel_stock['Close'] / intel_stock['Close'].iloc[0]) * 100

    plt.figure(figsize=(15, 10))
    plt.plot(amd_stock['Date'], amd_stock['relative_performance'],
             label='AMD', color='red')
    plt.plot(intel_stock['Date'], intel_stock['relative_performance'],
             label='Intel', color='blue')
    plt.title('AMD vs Intel Relative Stock Performance')
    plt.xlabel('Date')
    plt.ylabel('Relative Performance (Base 100)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'amd_intel_performance.png'))
    plt.close()

    cpu_keywords = ['cpu', 'processor', 'ryzen',
                    'core', 'x3d', 'performance', 'benchmark']

    def is_cpu_related(text: str) -> bool:
        """Check if text contains CPU-related keywords"""
        if pd.isna(text):
            return False
        return any(keyword in str(text).lower() for keyword in cpu_keywords)

    amd_cpu_news = news_data['AMD'][news_data['AMD']['title'].apply(is_cpu_related) |
                                    news_data['AMD']['summary'].apply(is_cpu_related)].copy()
    intel_cpu_news = news_data['INTC'][news_data['INTC']['title'].apply(is_cpu_related) |
                                       news_data['INTC']['summary'].apply(is_cpu_related)].copy()

    amd_cpu_news['date'] = pd.to_datetime(amd_cpu_news['date'])
    intel_cpu_news['date'] = pd.to_datetime(intel_cpu_news['date'])

    plt.figure(figsize=(15, 10))
    plt.scatter(amd_cpu_news['date'], [1] * len(amd_cpu_news), label='AMD CPU News',
                color='red', alpha=0.5, s=100)
    plt.scatter(intel_cpu_news['date'], [0] * len(intel_cpu_news), label='Intel CPU News',
                color='blue', alpha=0.5, s=100)
    plt.title('Timeline of CPU-Related News')
    plt.xlabel('Date')
    plt.yticks([0, 1], ['Intel', 'AMD'])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cpu_news_timeline.png'))
    plt.close()

    amd_cpu_news.to_csv(os.path.join(
        output_dir, 'amd_cpu_news.csv'), index=False)
    intel_cpu_news.to_csv(os.path.join(
        output_dir, 'intel_cpu_news.csv'), index=False)

    def calculate_period_performance(start_date: str, end_date: str) -> Tuple[float, float]:
        amd_perf = amd_stock[(amd_stock['Date'] >= start_date) &
                             (amd_stock['Date'] <= end_date)]['relative_performance']
        intel_perf = intel_stock[(intel_stock['Date'] >= start_date) &
                                 (intel_stock['Date'] <= end_date)]['relative_performance']
        return (
            ((amd_perf.iloc[-1] - amd_perf.iloc[0]) / amd_perf.iloc[0]) * 100,
            ((intel_perf.iloc[-1] - intel_perf.iloc[0]) /
             intel_perf.iloc[0]) * 100
        )

    periods = {
        'Overall': ('2019-01-01', '2023-12-31'),
        'Ryzen 7000 Launch': ('2022-09-01', '2022-12-31'),
        'X3D Launch': ('2023-02-01', '2023-05-31')
    }

    with open(os.path.join(output_dir, 'performance_analysis.txt'), 'w') as f:
        f.write("AMD vs Intel Performance Analysis\n")
        f.write("================================\n\n")

        for period_name, (start_date, end_date) in periods.items():
            amd_change, intel_change = calculate_period_performance(
                start_date, end_date)
            f.write(f"{period_name} ({start_date} to {end_date}):\n")
            f.write(f"AMD Performance Change: {amd_change:.2f}%\n")
            f.write(f"Intel Performance Change: {intel_change:.2f}%\n")
            f.write(
                f"Relative Difference: {amd_change - intel_change:.2f}%\n\n")

        f.write("\nCPU-Related News Summary:\n")
        f.write("========================\n")
        f.write(f"Total AMD CPU-related news: {len(amd_cpu_news)}\n")
        f.write(f"Total Intel CPU-related news: {len(intel_cpu_news)}\n\n")

        f.write("Recent AMD CPU News:\n")
        for _, row in amd_cpu_news.nlargest(5, 'date').iterrows():
            f.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['title']}\n")

        f.write("\nRecent Intel CPU News:\n")
        for _, row in intel_cpu_news.nlargest(5, 'date').iterrows():
            f.write(f"{row['date'].strftime('%Y-%m-%d')}: {row['title']}\n")
