# forecast.py

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

class Forecast:

    @staticmethod
    def prepare_data(df, column="Adj Close", train_end="2024-12-31"):
        """
        Split time series into train and test sets chronologically.
        Assumes input data is already cleaned and frequency-aligned.
        """
        df = df.copy()
        df = df[[column]]

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Chronological split
        train = df[df.index <= train_end]
        test  = df[df.index >  train_end]

        return train, test


    @staticmethod
    def arima_forecast(train, test, seasonal=False, m=1):
        """
        Fit ARIMA/SARIMA using auto_arima.
        Returns fitted model and forecast DataFrame.
        """

        # Convert DataFrame → Series
        y_train = train.iloc[:, 0]

        stepwise_model = auto_arima(
            y_train,
            seasonal=seasonal,
            m=m,
            trace=False,
            error_action="ignore",
            suppress_warnings=True
        )

        model = ARIMA(
            y_train,
            order=stepwise_model.order,
            seasonal_order=getattr(stepwise_model, "seasonal_order", (0,0,0,0))
        )

        model_fit = model.fit()

        forecast_values = model_fit.forecast(steps=len(test))

        forecast = pd.DataFrame(
            forecast_values.values,
            index=test.index,
            columns=["Forecast"]
        )

        return model_fit, forecast


    @staticmethod
    def lstm_forecast(train, test, window_size=60, epochs=50, batch_size=32, neurons=50, return_model=False):
        """
        Build and train an LSTM to predict next-day prices.
        Returns forecast DataFrame indexed like test.
        Optionally returns the trained model and scaler for future forecasting.
        """
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train = scaler.fit_transform(train)
        scaled_test = scaler.transform(test)
        
        # Prepare sequences
        X_train, y_train = Forecast.create_sequences(scaled_train, window_size)
        X_test, y_test   = Forecast.create_sequences(
            np.concatenate([scaled_train[-window_size:], scaled_test]),
            window_size
        )
        
        # Build model
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], 1)))
        model.add(LSTM(neurons, return_sequences=True))
        model.add(LSTM(neurons))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model with early stopping
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=0.1,
            callbacks=[EarlyStopping(patience=5, monitor='loss')]
        )
        
        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        forecast_df = pd.DataFrame(predictions, index=test.index, columns=["Forecast"])
        
        if return_model:
            return forecast_df, model, scaler
        else:
            return forecast_df


    @staticmethod
    def create_sequences(data, window_size):
        """
        Helper function to create sequences for LSTM.
        """
        X, y = [], []
        for i in range(window_size, len(data)):
            X.append(data[i-window_size:i, 0])
            y.append(data[i, 0])
        X = np.array(X)
        y = np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        return X, y


    # Evaluation Metrics

    @staticmethod
    def evaluate_models(test, *forecasts):
        """
        Evaluate forecasts with MAE, RMSE, MAPE.
        Returns a DataFrame.
        """
        results = []
        true_values = test.iloc[:, 0]  # extract the series
        for i, forecast in enumerate(forecasts):
            # Align indices
            forecast_aligned = forecast.reindex(true_values.index)
            df_eval = pd.concat([true_values, forecast_aligned], axis=1).dropna()
            
            mae = np.mean(np.abs(df_eval.iloc[:, 0] - df_eval["Forecast"]))
            rmse = np.sqrt(np.mean((df_eval.iloc[:, 0] - df_eval["Forecast"])**2))
            mape = np.mean(np.abs((df_eval.iloc[:, 0] - df_eval["Forecast"]) / df_eval.iloc[:, 0])) * 100
            
            results.append([f"Model {i+1}", mae, rmse, mape])
            
        return pd.DataFrame(results, columns=["Model", "MAE", "RMSE", "MAPE"])

    # LSTM Multi-step Forecast
    
    @staticmethod
    def lstm_future_forecast(model, train, window_size=60, steps=126, neurons=50):
        """
        Generate multi-step forecast using a trained LSTM model.
        - model: trained Keras LSTM model
        - train: historical series (pd.DataFrame) used for scaling
        - window_size: look-back size for predictions
        - steps: number of future steps to forecast (default ~6 months of business days)
        """

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_train = scaler.fit_transform(train)

        # last known window
        last_window = scaled_train[-window_size:].flatten()
        future_preds = []

        for _ in range(steps):
            x_input = last_window.reshape((1, window_size, 1))
            pred = model.predict(x_input, verbose=0)
            future_preds.append(pred[0, 0])
            last_window = np.append(last_window[1:], pred[0, 0])

        future_preds = np.array(future_preds).reshape(-1, 1)
        future_preds = scaler.inverse_transform(future_preds).flatten()
        return future_preds
    
    @staticmethod
    def generate_future_index(last_date, steps, freq='B'):
        return pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq=freq)