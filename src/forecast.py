# forecast.py

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

class Forecast:

    @staticmethod
    def prepare_data(df, column="Adj Close", train_end="2024-12-31"):
        """
        Split time series into train and test sets chronologically.
        Ensures datetime index with business day frequency.
        """
        df = df.copy()
        df = df[[column]]

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df = df.reset_index().rename(columns={df.columns[0]: "Date"}).set_index("Date")
        df.index = pd.to_datetime(df.index)

        # Set business day frequency and fill missing values
        df = df.asfreq('B').ffill()

        # Split chronologically
        train = df[df.index <= train_end]
        test  = df[df.index >  train_end]
        return train, test

    # -----------------------
    # ARIMA/SARIMA Forecast
    # -----------------------
    @staticmethod
    def arima_forecast(train, test, seasonal=False, m=1):
        """
        Fit ARIMA/SARIMA using auto_arima to determine optimal parameters.
        Returns fitted ARIMA model and forecast DataFrame indexed like test.
        """
        # Auto ARIMA to find best params
        stepwise_model = auto_arima(
            train,
            seasonal=seasonal,
            m=m,
            trace=False,
            error_action="ignore",
            suppress_warnings=True
        )

        # Extract seasonal_order if exists, otherwise default
        seasonal_order = getattr(stepwise_model, 'seasonal_order', (0,0,0,0))
        
        # Fit ARIMA
        model = ARIMA(
            train,
            order=stepwise_model.order,
            seasonal_order=seasonal_order
        )
        model_fit = model.fit()
        
        # Forecast
        forecast_values = model_fit.forecast(len(test))
        forecast = pd.DataFrame(forecast_values, index=test.index, columns=["Forecast"])
        return model_fit, forecast

    # -----------------------
    # LSTM Forecast
    # -----------------------
    @staticmethod
    def lstm_forecast(train, test, window_size=60, epochs=50, batch_size=32, neurons=50):
        """
        Build and train an LSTM to predict next-day prices.
        Returns forecast DataFrame indexed like test.
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
        forecast = pd.DataFrame(predictions, index=test.index, columns=["Forecast"])
        return forecast

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

    # -----------------------
    # Evaluation Metrics
    # -----------------------
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
