import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def calculate_technical_indicators(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators from price data
    """
    df = price_data.copy()

    df['returns'] = df['Close'].pct_change()
    df['returns_5d'] = df['Close'].pct_change(periods=5)
    df['returns_10d'] = df['Close'].pct_change(periods=10)

    df['ma_5'] = df['Close'].rolling(window=5).mean()
    df['ma_10'] = df['Close'].rolling(window=10).mean()
    df['ma_20'] = df['Close'].rolling(window=20).mean()

    df['volatility_5d'] = df['returns'].rolling(window=5).std()
    df['volatility_10d'] = df['returns'].rolling(window=10).std()

    df['price_ma5_ratio'] = df['Close'] / df['ma_5']
    df['price_ma10_ratio'] = df['Close'] / df['ma_10']

    return df


def calculate_sentiment_features(sentiment_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate sentiment-based features
    """
    df = sentiment_data.copy()

    df['sentiment_ma3'] = df['headline_sentiment'].rolling(window=3).mean()
    df['sentiment_ma5'] = df['headline_sentiment'].rolling(window=5).mean()
    df['sentiment_ma10'] = df['headline_sentiment'].rolling(window=10).mean()

    df['sentiment_std3'] = df['headline_sentiment'].rolling(window=3).std()
    df['sentiment_std5'] = df['headline_sentiment'].rolling(window=5).std()

    df['news_volume_ma3'] = df['news_count'].rolling(window=3).mean()
    df['news_volume_ma5'] = df['news_count'].rolling(window=5).mean()

    df['sentiment_momentum'] = df['sentiment_ma3'] - df['sentiment_ma5']

    return df


def prepare_features_for_prediction(
    price_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    lookback_period: int = 10
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features for prediction models
    """
    print("\nFeature preparation debug info:")
    print(f"Price data shape: {price_data.shape}")
    print(f"Sentiment data shape: {sentiment_data.shape}")

    # Ensure date columns are properly formatted
    price_data['Date'] = pd.to_datetime(price_data['Date']).dt.date
    sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date

    # Calculate technical indicators
    price_features = calculate_technical_indicators(price_data)
    print(
        f"Price features shape after technical indicators: {price_features.shape}")

    # Calculate sentiment features
    sentiment_features = calculate_sentiment_features(sentiment_data)
    print(
        f"Sentiment features shape after calculation: {sentiment_features.shape}")

    features = pd.merge(
        price_features,
        sentiment_features,
        left_on='Date',
        right_on='date',
        how='left'
    )
    print(f"Features shape after merging: {features.shape}")

    features = features.fillna(method='ffill').fillna(method='bfill')

    feature_columns = [
        'returns', 'returns_5d', 'returns_10d',
        'volatility_5d', 'volatility_10d',
        'price_ma5_ratio', 'price_ma10_ratio',
        'sentiment_ma3', 'sentiment_ma5', 'sentiment_ma10',
        'sentiment_std3', 'sentiment_std5',
        'news_volume_ma3', 'news_volume_ma5',
        'sentiment_momentum'
    ]

    missing_columns = [
        col for col in feature_columns if col not in features.columns]
    if missing_columns:
        print(f"\nWarning: Missing feature columns: {missing_columns}")
        print(f"Available columns: {features.columns.tolist()}")
        raise ValueError(
            f"Missing required feature columns: {missing_columns}")

    X = features[feature_columns]
    print(f"\nSelected features: {feature_columns}")
    print(f"Feature matrix shape: {X.shape}")

    target = features['Close'].pct_change().shift(-1)

    X = X[:-1]
    target = target[:-1]

    print(f"Final features shape: {X.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Target value range: [{target.min():.4f}, {target.max():.4f}]")

    target = (target > 0).astype(int)
    print(f"Binary target distribution:\n{target.value_counts()}")

    return X, target


def create_sequences(
    X: pd.DataFrame,
    y: pd.Series,
    sequence_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM model
    """
    X_seq = []
    y_seq = []

    for i in range(len(X) - sequence_length):
        X_seq.append(X.iloc[i:(i + sequence_length)].values)
        y_seq.append(y.iloc[i + sequence_length])

    return np.array(X_seq), np.array(y_seq)
