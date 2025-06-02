import pandas as pd
import matplotlib.pyplot as plt
import os


def create_news_distribution_chart():
    # Get all news files
    data_dir = 'data'
    news_files = [f for f in os.listdir(data_dir) if f.endswith('_news.csv')]

    # Count articles per ticker
    ticker_counts = {}
    for file in news_files:
        ticker = file.split('_')[0]
        df = pd.read_csv(os.path.join(data_dir, file))
        ticker_counts[ticker] = len(df)

    # Create pie chart
    plt.figure(figsize=(10, 8))
    plt.pie(ticker_counts.values(),
            labels=ticker_counts.keys(),
            autopct='%1.1f%%',
            startangle=90,
            shadow=True)

    plt.title('Distribution of News Articles by Ticker')
    # Equal aspect ratio ensures that pie is drawn as a circle
    plt.axis('equal')

    # Save the plot
    plt.savefig('results/visualizations/news_distribution.png')
    plt.close()


if __name__ == "__main__":
    create_news_distribution_chart()
