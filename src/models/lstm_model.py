import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pandas as pd
from typing import Dict, List, Tuple, Union


class StockLSTM:
    def __init__(self, sequence_length: int = 10, use_bidirectional: bool = True):
        """
        Initialize LSTM model for stock prediction using sentiment data

        Args:
            sequence_length: Number of time steps to look back
            use_bidirectional: Whether to use bidirectional LSTM
        """
        self.sequence_length = sequence_length
        self.use_bidirectional = use_bidirectional
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))

    def create_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Create LSTM model architecture

        Args:
            input_shape: Shape of input data (sequence_length, features)
        """
        model = Sequential()

        # First LSTM layer
        if self.use_bidirectional:
            model.add(Bidirectional(LSTM(units=64, return_sequences=True),
                                    input_shape=input_shape))
        else:
            model.add(LSTM(units=64, return_sequences=True,
                           input_shape=input_shape))

        model.add(Dropout(0.2))

        # Second LSTM layer
        if self.use_bidirectional:
            model.add(Bidirectional(LSTM(units=32, return_sequences=False)))
        else:
            model.add(LSTM(units=32, return_sequences=False))

        model.add(Dropout(0.2))

        # Dense layers
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=16, activation='relu'))
        model.add(Dense(units=1))

        # Huber loss is more robust to outliers
        model.compile(optimizer='adam', loss='huber')

        self.model = model
        return model

    def prepare_data(self, X: pd.DataFrame, y: Union[pd.Series, None] = None,
                     is_training: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare data for LSTM model, handling both sentiment and price features

        Args:
            X: DataFrame containing features (sentiment and price data)
            y: Target values (returns)
            is_training: Whether this is for training (True) or prediction (False)
        """
        if is_training:
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = self.feature_scaler.transform(X)

        # Create sequences
        X_seq = []
        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:(i + self.sequence_length)])
        X_seq = np.array(X_seq)

        if y is not None:
            y_seq = y[self.sequence_length:].values
            return X_seq, y_seq
        return X_seq

    def train(self, X: pd.DataFrame, y: pd.Series, epochs: int = 100,
              batch_size: int = 32, validation_split: float = 0.2) -> Dict:
        """
        Train the LSTM model

        Args:
            X: Features DataFrame
            y: Target Series
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data to use for validation
        """
        X_seq, y_seq = self.prepare_data(X, y)
        if self.model is None:
            self.create_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))

        history = self.model.fit(
            X_seq, y_seq,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        # Calculate training metrics
        train_size = int(len(X_seq) * (1 - validation_split))
        X_train, X_val = X_seq[:train_size], X_seq[train_size:]
        y_train, y_val = y_seq[:train_size], y_seq[train_size:]

        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)

        metrics = {
            'train_metrics': self.evaluate(y_train, train_pred),
            'val_metrics': self.evaluate(y_val, val_pred),
            'history': history.history
        }

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained model"""
        X_seq = self.prepare_data(X, is_training=False)
        predictions = self.model.predict(X_seq)
        return predictions

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate various metrics including MSE, RMSE, MAE, and KS test

        Args:
            y_true: True values
            y_pred: Predicted values
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(y_true, y_pred)

        # Direction accuracy (for returns)
        direction_accuracy = np.mean(np.sign(y_true) == np.sign(y_pred))

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'KS_statistic': ks_statistic,
            'p_value': p_value,
            'direction_accuracy': direction_accuracy
        }

    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Calculate feature importance using permutation importance

        Args:
            X: Features DataFrame
            y: Target Series
        """
        X_seq, y_seq = self.prepare_data(X, y)
        base_score = self.evaluate(y_seq, self.predict(X))['MSE']

        importance = {}
        for feature in X.columns:
            X_permuted = X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            permuted_score = self.evaluate(
                y_seq, self.predict(X_permuted))['MSE']
            importance[feature] = permuted_score - base_score

        return importance
