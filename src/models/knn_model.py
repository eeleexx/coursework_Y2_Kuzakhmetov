from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy import stats
import pandas as pd


class StockKNN:
    def __init__(self, n_neighbors=5):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def prepare_data(self, X, y=None, is_training=True):
        """Prepare data for KNN model. Scales X features."""
        # Ensure X is a DataFrame for .values access
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X)  # Convert if numpy array or list of lists
        else:
            X_df = X

        if X_df.empty:
            # Handle empty DataFrame case appropriately for your logic
            # For example, return empty scaled data or raise error
            if y is not None:
                return X_df, y
            return X_df

        if is_training:
            X_scaled = self.scaler.fit_transform(X_df.values)
        else:
            X_scaled = self.scaler.transform(X_df.values)

        # Return as DataFrame to maintain column names if needed later, though KNN itself uses numpy arrays
        X_scaled_df = pd.DataFrame(
            X_scaled, columns=X_df.columns, index=X_df.index)

        if y is not None:
            return X_scaled_df, y
        return X_scaled_df

    def train(self, X_prepared, y):
        """Train the KNN model. Assumes X_prepared is already scaled."""
        # X_prepared is already scaled by prepare_data
        self.model.fit(X_prepared.values,
                       y.values if isinstance(y, pd.Series) else y)

    def predict(self, X_prepared):
        """Make predictions. Assumes X_prepared is already scaled."""
        # X_prepared is already scaled by prepare_data
        return self.model.predict(X_prepared.values)

    def evaluate(self, y_true, y_pred):
        """Calculate various metrics including Kolmogorov-Smirnov test"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error

        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        # Kolmogorov-Smirnov test
        ks_statistic, p_value = stats.ks_2samp(y_true, y_pred)

        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'KS_statistic': ks_statistic,
            'p_value': p_value
        }
