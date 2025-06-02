from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict, Tuple, Union
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split


class StockKNN:
    def __init__(self, n_neighbors: int = 5, prediction_type: str = 'regression'):
        """
        Initialize KNN model for stock prediction

        Args:
            n_neighbors: Number of neighbors to use
            prediction_type: Either 'regression' for returns or 'classification' for direction
        """
        self.n_neighbors = n_neighbors
        self.prediction_type = prediction_type
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.lookback = None
        self.train_size = None

    def create_sequences(self, data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series prediction
        """
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i:i + lookback])
            y.append(data[i + lookback, 0])  # Predict next day's price
        return np.array(X), np.array(y)

    def prepare_data(self, data: pd.DataFrame, lookback: int = 10, train_size: float = 0.8):
        """
        Prepare data for time series prediction
        """
        self.lookback = lookback
        self.train_size = train_size

        train_data, test_data = train_test_split(
            data, test_size=1 - train_size, shuffle=False)

        self.scaler.fit(train_data)
        train_scaled = self.scaler.transform(train_data)
        test_scaled = self.scaler.transform(test_data)

        self.X_train, self.y_train = self.create_sequences(
            train_scaled, lookback)
        self.X_test, self.y_test = self.create_sequences(test_scaled, lookback)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], -1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], -1))

    def fit(self, data: pd.DataFrame, lookback: int = 10, train_size: float = 0.8):
        """
        Train the KNN model
        """
        self.prepare_data(data, lookback, train_size)
        self.model.fit(self.X_train, self.y_train)

        y_train_pred = self.model.predict(self.X_train)
        train_metrics = {
            'MSE': mean_squared_error(self.y_train, y_train_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'MAE': mean_absolute_error(self.y_train, y_train_pred),
            'R2': r2_score(self.y_train, y_train_pred)
        }

        return train_metrics

    def predict(self, X: Union[pd.DataFrame, np.ndarray] = None) -> np.ndarray:
        """
        Make predictions. If X is None, predict on test set.
        """
        if X is None:
            return self.model.predict(self.X_test)
        return self.model.predict(X)

    def evaluate(self, y_true: Union[pd.Series, np.ndarray] = None,
                 y_pred: Union[pd.Series, np.ndarray] = None) -> Dict:
        """
        Evaluate model performance
        """
        if y_true is None:
            y_true = self.y_test
        if y_pred is None:
            y_pred = self.predict()

        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred)
        }

        return metrics

    def plot_multiple_graphs(self, output_dir: str, symbol: str):
        """
        Plot actual vs predicted values for both training and test sets
        """
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        ax1.plot(self.y_train, label='Actual', color='blue')
        ax1.plot(y_train_pred, label='Predicted', color='red', linestyle='--')
        ax1.legend()

        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)

        ax1.set_title(
            f"Training Data (80% of data)\n"
            f"RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}"
        )
        ax1.set_xlabel("Time Steps")
        ax1.set_ylabel("Price")

        ax2.plot(self.y_test, label='Actual', color='blue')
        ax2.plot(y_test_pred, label='Predicted', color='red', linestyle='--')
        ax2.legend()

        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        test_mae = mean_absolute_error(self.y_test, y_test_pred)

        ax2.set_title(
            f"Test Data (20% of data)\n"
            f"RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}"
        )
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Price")

        fig.suptitle(
            f"KNN Model Predictions for {symbol}\n"
            f"Lookback Window: {self.lookback} days",
            fontsize=14,
            y=1.02
        )

        plt.tight_layout()

        output_path = os.path.join(output_dir, f'{symbol}_knn_predictions.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        return {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }

    def save_model(self, model_dir: str, symbol: str):
        """
        Save the trained model and scaler
        """
        model_path = os.path.join(model_dir, f'{symbol}_knn_model.joblib')
        scaler_path = os.path.join(model_dir, f'{symbol}_knn_scaler.joblib')

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load_model(self, model_dir: str, symbol: str):
        """
        Load a trained model and scaler
        """
        model_path = os.path.join(model_dir, f'{symbol}_knn_model.joblib')
        scaler_path = os.path.join(model_dir, f'{symbol}_knn_scaler.joblib')

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def plot_feature_importance(self, X: pd.DataFrame, y: pd.Series,
                                output_dir: str, symbol: str):
        """
        Plot feature importance based on permutation
        """
        feature_importance = {}
        baseline_score = self.evaluate(y, self.predict(X))

        if self.prediction_type == 'regression':
            metric_key = 'R2'
            baseline_metric = baseline_score['R2']
        else:
            metric_key = 'Accuracy'
            baseline_metric = baseline_score['Accuracy']

        for feature in X.columns:
            X_permuted = X.copy()
            X_permuted[feature] = np.random.permutation(X_permuted[feature])
            permuted_score = self.evaluate(
                y, self.predict(X_permuted))[metric_key]
            feature_importance[feature] = baseline_metric - permuted_score

        plt.figure(figsize=(12, 6))
        importance_df = pd.Series(
            feature_importance).sort_values(ascending=True)

        sns.barplot(x=importance_df.values, y=importance_df.index)
        plt.title(f'Feature Importance for {symbol} - KNN Model')
        plt.xlabel('Importance Score')

        plt.tight_layout()
        plt.savefig(os.path.join(
            output_dir, f'{symbol}_knn_feature_importance.png'))
        plt.close()

        return feature_importance

    def plot_predictions(self, X: pd.DataFrame, y: pd.Series,
                         output_dir: str, symbol: str):
        """
        Plot actual vs predicted values
        """
        y_pred = self.predict(X)

        plt.figure(figsize=(15, 6))

        if self.prediction_type == 'regression':
            plt.scatter(y, y_pred, alpha=0.5)
            plt.plot([y.min(), y.max()], [y.min(), y.max()],
                     'r--', lw=2)  # Perfect prediction line
            plt.xlabel('Actual Returns')
            plt.ylabel('Predicted Returns')
        else:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

        plt.title(f'{symbol} - KNN Predictions vs Actual')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{symbol}_knn_predictions.png'))
        predictions_dir = os.path.join(
            os.path.dirname(output_dir), 'predictions')
        if os.path.exists(predictions_dir):
            plt.savefig(os.path.join(predictions_dir,
                        f'{symbol}_knn_predictions.png'))
        plt.close()
