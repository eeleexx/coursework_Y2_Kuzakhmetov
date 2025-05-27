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

        if prediction_type == 'regression':
            self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        else:
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

        self.scaler = StandardScaler()
        self.feature_names = None

    def prepare_data(self, X: pd.DataFrame, y: Union[pd.Series, None] = None,
                     is_training: bool = True) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
        """
        Prepare data for KNN model by scaling features
        """
        # Drop non-feature columns
        feature_columns = [col for col in X.columns if col not in [
            'Date', 'date', 'symbol', 'Close', 'Volume']]
        X = X[feature_columns]

        if is_training:
            self.feature_names = X.columns
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=self.feature_names,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=self.feature_names,
                index=X.index
            )

        if y is not None:
            print("\nTarget preparation debug:")
            print(f"Original y type: {type(y)}")
            print(f"Original y dtype: {y.dtype}")
            print(f"Original y unique values: {y.unique()}")

            # Ensure y is binary for classification
            if self.prediction_type == 'classification':
                y = y.astype(int)
                print(f"Converted y dtype: {y.dtype}")
                print(f"Converted y unique values: {y.unique()}")

            return X_scaled, y

        return X_scaled

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the KNN model and return performance metrics
        """
        X_scaled, y_processed = self.prepare_data(X, y, is_training=True)
        self.model.fit(X_scaled, y_processed)

        # Make predictions on training data
        y_pred = self.predict(X)

        # Calculate metrics
        metrics = self.evaluate(y_processed, y_pred)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        """
        X_scaled = self.prepare_data(X, is_training=False)
        predictions = self.model.predict(X_scaled)

        print("\nPrediction debug:")
        print(f"Predictions type: {type(predictions)}")
        print(f"Predictions dtype: {predictions.dtype}")
        print(f"Predictions unique values: {np.unique(predictions)}")

        return predictions

    def evaluate(self, y_true: Union[pd.Series, np.ndarray],
                 y_pred: Union[pd.Series, np.ndarray]) -> Dict:
        """
        Evaluate model performance
        """
        print("\nEvaluation debug:")
        print(f"y_true type: {type(y_true)}")
        print(f"y_true dtype: {y_true.dtype}")
        print(f"y_true unique values: {np.unique(y_true)}")
        print(f"y_pred type: {type(y_pred)}")
        print(f"y_pred dtype: {y_pred.dtype}")
        print(f"y_pred unique values: {np.unique(y_pred)}")

        # Convert to numpy arrays if needed
        if isinstance(y_true, pd.Series):
            y_true = y_true.values
        if isinstance(y_pred, pd.Series):
            y_pred = y_pred.values

        # Ensure both are integers for classification
        if self.prediction_type == 'classification':
            y_true = y_true.astype(int)
            y_pred = y_pred.astype(int)

            print("\nFinal conversion check:")
            print(f"Final y_true dtype: {y_true.dtype}")
            print(f"Final y_pred dtype: {y_pred.dtype}")
            print(f"Final y_true unique values: {np.unique(y_true)}")
            print(f"Final y_pred unique values: {np.unique(y_pred)}")

            metrics = {
                'Accuracy': accuracy_score(y_true, y_pred),
                'Precision': precision_score(y_true, y_pred, zero_division=0),
                'Recall': recall_score(y_true, y_pred, zero_division=0),
                'F1': f1_score(y_true, y_pred, zero_division=0)
            }
        else:
            metrics = {
                'MSE': mean_squared_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'MAE': mean_absolute_error(y_true, y_pred),
                'R2': r2_score(y_true, y_pred)
            }

        return metrics

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

        # Plot feature importance
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
        plt.close()
