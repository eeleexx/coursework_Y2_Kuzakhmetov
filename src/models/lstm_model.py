import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from typing import List, Optional, Dict, Tuple
import os
import matplotlib.pyplot as plt

RANDOM_STATE = 42


class Model:
    def __init__(self, df, filters=[]):
        self.data = df[filters]
        self.filters = filters
        self.X_train = None
        self.y_train = None
        self.X_Test = None
        self.y_test = None
        self.model = None
        self.history = None
        self.scaler = None
        self.test_mae = 0
        self.test_rmse = 0
        self.train_mae = 0
        self.train_rmse = 0
        self.train_predict = None
        self.test_predict = None

    def fit(self):
        pass

    def draw_train_graph(self, ticker=None, ax=None):
        Xt = self.model.predict(self.X_train)
        Xt = Xt.flatten()

        df_actual_keys = ["Actual"]
        if len(self.filters) > 1:
            df_actual_keys += ['null_val']
        df_actual_keys += [f"null_val{i}" for i in range(
            1, len(self.filters) - 1)]
        df_actual_data = {"Actual": self.y_train}
        for _key in df_actual_keys[1:]:
            df_actual_data[_key] = [0] * len(self.y_train)

        df_predicted_keys = ["Predicted"]
        if len(self.filters) > 1:
            df_predicted_keys.append('null_val')
        df_predicted_keys += [
            f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        df_predicted_data = {"Predicted": Xt}
        for _key in df_predicted_keys[1:]:
            df_predicted_data[_key] = [0] * len(Xt)

        df_actual = pd.DataFrame(df_actual_data)
        df_actual[df_actual_keys] = self.scaler.inverse_transform(
            df_actual[df_actual_keys])

        df_predicted = pd.DataFrame(df_predicted_data)
        df_predicted[df_predicted_keys] = self.scaler.inverse_transform(
            df_predicted[df_predicted_keys])
        self.train_predict = df_predicted.Predicted

        ax.plot(df_actual.Actual, label="Actual", color='blue')
        ax.plot(self.train_predict, label="Predicted",
                color='red', linestyle='--')
        ax.legend()

        self.train_rmse = math.sqrt(mean_squared_error(
            df_actual.Actual, self.train_predict))
        self.train_mae = mean_absolute_error(
            df_actual.Actual, self.train_predict)

        ax.set_title(
            f"Training Data (80% of data)\n"
            f"RMSE: {self.train_rmse:.2f}, MAE: {self.train_mae:.2f}"
        )
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Price")

        print(
            f"[{self.filters}] [{self.__class__.__name__}] Train RMSE =", self.train_rmse)
        print(
            f"[{self.filters}] [{self.__class__.__name__}] Train MAE =", self.train_mae)

    def draw_test_graph(self, ticker=None, ax=None):
        Xt = self.model.predict(self.X_Test)
        Xt = Xt.flatten()

        df_actual_keys = ["Actual"]
        df_actual_data = {"Actual": self.y_test}
        if len(self.filters) > 1:
            df_actual_keys.append('null_val')
        df_actual_keys += [f"null_val{i}" for i in range(
            1, len(self.filters) - 1)]
        for _key in df_actual_keys[1:]:
            df_actual_data[_key] = [0] * len(self.y_test)

        df_predicted_keys = ["Predicted"]
        df_predicted_data = {"Predicted": Xt}

        if len(self.filters) > 1:
            df_predicted_keys.append('null_val')
        df_predicted_keys += [
            f"null_val{i}" for i in range(1, len(self.filters) - 1)]
        for _key in df_predicted_keys[1:]:
            df_predicted_data[_key] = [0] * len(Xt)

        df_actual = pd.DataFrame(df_actual_data)
        df_actual[df_actual_keys] = self.scaler.inverse_transform(
            df_actual[df_actual_keys])

        df_predicted = pd.DataFrame(df_predicted_data)
        df_predicted[df_predicted_keys] = self.scaler.inverse_transform(
            df_predicted[df_predicted_keys])
        self.test_predict = df_predicted.Predicted

        ax.plot(df_actual.Actual, label="Actual", color='blue')
        ax.plot(self.test_predict, label="Predicted",
                color='red', linestyle='--')
        ax.legend()

        self.test_rmse = math.sqrt(mean_squared_error(
            df_actual.Actual, self.test_predict))
        self.test_mae = mean_absolute_error(
            df_actual.Actual, self.test_predict)

        ax.set_title(
            f"Test Data (20% of data)\n"
            f"RMSE: {self.test_rmse:.2f}, MAE: {self.test_mae:.2f}"
        )
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Price")

        print(
            f"[{self.filters}] [{self.__class__.__name__}] Test RMSE =", self.test_rmse)
        print(
            f"[{self.filters}] [{self.__class__.__name__}] Test MAE =", self.test_mae)

    def plot_multiple_graphs(self, ticker=None, output_dir=None):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

        self.draw_train_graph(ticker, axs[0])
        self.draw_test_graph(ticker, axs[1])

        fig.suptitle(
            f"LSTM Model Predictions for {ticker}\n"
            f"Features: {', '.join(self.filters)}",
            fontsize=14,
            y=1.02
        )

        plt.tight_layout()

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(
            output_dir, f'{ticker}_{self.__class__.__name__}_{"_".join(self.filters)}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def export(self, metric_params: Optional[List] = None):
        if not metric_params:
            result = {
                "test_rmse": float(self.test_rmse),
                "test_mae": float(self.test_mae),
                "train_rmse": float(self.train_rmse),
                "train_mae": float(self.train_mae),
                "test_predict": list(map(float, self.test_predict.tolist())),
            }
            return result
        result = {}
        for metric_param in metric_params:
            result[metric_param] = getattr(self, metric_param)
        return result


class StockLSTM(Model):
    def __init__(self, df, filters):
        super().__init__(df, filters=filters)

    def process_data(self, lookback, train_size, scaler=StandardScaler):
        data = self.data.copy()

        train_data, test_data = train_test_split(
            data, test_size=1 - train_size, shuffle=False)

        self.scaler = scaler()
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)

        X_train, y_train = [], []
        for i in range(len(train_data) - lookback - 1):
            X_train.append(train_data[i:i + lookback])
            y_train.append(train_data[i + lookback, 0])
        self.X_train, self.y_train = np.array(X_train), np.array(y_train)

        X_test, y_test = [], []
        for i in range(len(test_data) - lookback - 1):
            X_test.append(test_data[i:i + lookback])
            y_test.append(test_data[i + lookback, 0])
        self.X_Test, self.y_test = np.array(X_test), np.array(y_test)

        self.X_train = self.X_train.reshape(
            (self.X_train.shape[0], self.X_train.shape[1], len(self.filters)))
        self.X_Test = self.X_Test.reshape(
            (self.X_Test.shape[0], self.X_Test.shape[1], len(self.filters)))

    def fit(self, lookback=10, train_size=0.8, scaler=StandardScaler, epochs=300, model_dir=None, ticker=None):
        self.process_data(lookback, train_size, scaler)
        tf.keras.backend.clear_session()
        tf.random.set_seed(RANDOM_STATE)
        np.random.seed(RANDOM_STATE)

        inputs = Input(shape=(lookback, len(self.filters)))
        x = LSTM(units=128, return_sequences=True)(inputs)
        x = Dropout(0.1)(x)
        x = LSTM(64)(x)
        x = Dropout(0.1)(x)
        outputs = Dense(units=1)(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        callbacks_list = [
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=25,
                restore_best_weights=True
            )
        ]

        if model_dir and ticker:
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f'{ticker}_lstm_model.keras')
            callbacks_list.append(
                ModelCheckpoint(
                    filepath=model_path,
                    monitor='val_loss',
                    save_best_only=True,
                    mode='min'
                )
            )

        self.history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            validation_data=(self.X_Test, self.y_test),
            shuffle=False,
            callbacks=callbacks_list,
            verbose=1
        )

        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_Test)

        self.train_rmse = np.sqrt(
            mean_squared_error(self.y_train, y_train_pred))
        self.train_mae = mean_absolute_error(self.y_train, y_train_pred)
        self.test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        self.test_mae = mean_absolute_error(self.y_test, y_test_pred)

        print("\nTraining Summary:")
        print(f"Final training loss: {self.history.history['loss'][-1]:.4f}")
        print(
            f"Final validation loss: {self.history.history['val_loss'][-1]:.4f}")
        print(f"Number of epochs trained: {len(self.history.history['loss'])}")
        print(f"Training RMSE: {self.train_rmse:.4f}")
        print(f"Training MAE: {self.train_mae:.4f}")
        print(f"Test RMSE: {self.test_rmse:.4f}")
        print(f"Test MAE: {self.test_mae:.4f}")

        return self.history

    def predict(self, X: np.ndarray = None) -> np.ndarray:
        """
        Make predictions
        """
        if X is None:
            return self.model.predict(self.X_Test)
        return self.model.predict(X)

    def evaluate(self) -> Dict:
        """
        Evaluate model performance
        """
        y_pred = self.predict()
        y_true = self.y_test

        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'train_rmse': self.train_rmse,
            'train_mae': self.train_mae,
            'test_rmse': self.test_rmse,
            'test_mae': self.test_mae
        }

        return metrics

    def plot_multiple_graphs(self, ticker: str, output_dir: str):
        """
        Plot actual vs predicted values for both training and test sets
        """
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_Test)

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
            f"LSTM Model Predictions for {ticker}\n"
            f"Features: {', '.join(self.filters)}",
            fontsize=14,
            y=1.02
        )

        plt.tight_layout()

        output_path = os.path.join(
            output_dir, f'{ticker}_lstm_predictions.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

        return {
            'train_rmse': train_rmse,
            'train_mae': train_mae,
            'test_rmse': test_rmse,
            'test_mae': test_mae
        }

    def export(self) -> Dict:
        """
        Export model metrics and predictions
        """
        y_pred = self.predict()
        y_true = self.y_test

        metrics = self.evaluate()
        metrics.update({
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, self.model.predict(self.X_train))),
            'train_mae': mean_absolute_error(self.y_train, self.model.predict(self.X_train)),
            'test_predictions': y_pred.tolist(),
            'test_actual': y_true.tolist()
        })

        return metrics
