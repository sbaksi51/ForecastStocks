"""
Advanced Time Series Models for Stock Price Prediction
Includes LSTM, GRU, Transformer architectures, and ensemble methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    MultiHeadAttention, LayerNormalization, Conv1D,
    GlobalAveragePooling1D, Input, Concatenate
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# PyTorch for advanced models
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Time Series
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# Utilities
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os

logger = logging.getLogger(__name__)

class TimeSeriesPredictor:
    """Advanced time series prediction using multiple architectures"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.model_dir = config.get('model_dir', 'models/timeseries/')
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        torch.manual_seed(42)
    
    def prepare_sequences(self, data: pd.DataFrame, lookback: int = 60, 
                         features: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for time series models"""
        
        if features is None:
            features = ['open', 'high', 'low', 'close', 'volume']
        
        # Ensure all features exist
        available_features = [f for f in features if f in data.columns]
        feature_data = data[available_features].values
        
        # Scale data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i])
            y.append(scaled_data[i, available_features.index('close')])  # Predict close price
        
        return np.array(X), np.array(y), scaler
    
    def build_lstm_model(self, input_shape: Tuple) -> Model:
        """Build advanced LSTM model"""
        
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            BatchNormalization(),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',  # More robust to outliers
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_gru_model(self, input_shape: Tuple) -> Model:
        """Build GRU model"""
        
        model = Sequential([
            GRU(128, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            BatchNormalization(),
            
            GRU(64, return_sequences=True),
            Dropout(0.2),
            BatchNormalization(),
            
            GRU(32, return_sequences=False),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_transformer_model(self, input_shape: Tuple) -> Model:
        """Build Transformer model for time series"""
        
        inputs = Input(shape=input_shape)
        
        # Positional encoding
        positions = tf.range(start=0, limit=input_shape[0], delta=1)
        position_embeddings = tf.keras.layers.Embedding(
            input_dim=input_shape[0], output_dim=input_shape[1]
        )(positions)
        
        x = inputs + position_embeddings
        
        # Multi-head attention blocks
        for _ in range(3):
            # Multi-head attention
            attn_output = MultiHeadAttention(
                num_heads=4, key_dim=input_shape[1] // 4
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # Feed forward
            ffn = Sequential([
                Dense(128, activation='relu'),
                Dense(input_shape[1])
            ])(x)
            x = LayerNormalization(epsilon=1e-6)(x + ffn)
        
        # Global pooling and output
        x = GlobalAveragePooling1D()(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='huber',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def build_cnn_lstm_model(self, input_shape: Tuple) -> Model:
        """Build hybrid CNN-LSTM model"""
        
        model = Sequential([
            # CNN layers for feature extraction
            Conv1D(64, kernel_size=3, activation='relu', input_shape=input_shape),
            Conv1D(32, kernel_size=3, activation='relu'),
            Dropout(0.2),
            
            # LSTM layers for sequence learning
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            # Dense layers
            Dense(25, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_ensemble(self, symbol: str, data: pd.DataFrame, 
                      lookback: int = 60, epochs: int = 50) -> Dict:
        """Train ensemble of models"""
        
        # Prepare data
        X, y, scaler = self.prepare_sequences(data, lookback)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        
        models = {
            'lstm': self.build_lstm_model((lookback, X.shape[2])),
            'gru': self.build_gru_model((lookback, X.shape[2])),
            'transformer': self.build_transformer_model((lookback, X.shape[2])),
            'cnn_lstm': self.build_cnn_lstm_model((lookback, X.shape[2]))
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name} model for {symbol}")
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=epochs,
                batch_size=32,
                callbacks=[early_stop, reduce_lr],
                verbose=0
            )
            
            # Evaluate
            test_loss = model.evaluate(X_test, y_test, verbose=0)
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{symbol}_{name}_model.h5")
            model.save(model_path)
            
            results[name] = {
                'model': model,
                'history': history.history,
                'test_loss': test_loss,
                'model_path': model_path
            }
            
            logger.info(f"{name} model - Test Loss: {test_loss}")
        
        # Save scaler
        scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")
        joblib.dump(scaler, scaler_path)
        
        # Store in memory
        self.models[symbol] = results
        self.scalers[symbol] = scaler
        
        return results
    
    def predict_ensemble(self, symbol: str, recent_data: pd.DataFrame, 
                        days_ahead: int = 5) -> Dict:
        """Make predictions using ensemble of models"""
        
        if symbol not in self.models:
            raise ValueError(f"No trained models found for {symbol}")
        
        # Prepare recent data
        lookback = 60
        X, _, scaler = self.prepare_sequences(recent_data, lookback)
        
        if len(X) == 0:
            raise ValueError("Insufficient data for prediction")
        
        # Get last sequence
        last_sequence = X[-1].reshape(1, lookback, X.shape[2])
        
        predictions = {}
        
        # Get predictions from each model
        for name, model_info in self.models[symbol].items():
            model = model_info['model']
            model_predictions = []
            
            current_sequence = last_sequence.copy()
            
            for _ in range(days_ahead):
                # Predict next value
                pred = model.predict(current_sequence, verbose=0)
                model_predictions.append(pred[0, 0])
                
                # Update sequence
                new_row = current_sequence[0, -1].copy()
                new_row[3] = pred[0, 0]  # Update close price
                current_sequence = np.append(
                    current_sequence[0, 1:], [new_row], axis=0
                ).reshape(1, lookback, X.shape[2])
            
            predictions[name] = model_predictions
        
        # Ensemble predictions (weighted average)
        weights = {
            'lstm': 0.3,
            'gru': 0.2,
            'transformer': 0.35,
            'cnn_lstm': 0.15
        }
        
        ensemble_predictions = []
        for i in range(days_ahead):
            weighted_pred = sum(
                predictions[model][i] * weights.get(model, 0.25)
                for model in predictions
            )
            ensemble_predictions.append(weighted_pred)
        
        # Inverse transform predictions
        current_price = recent_data['close'].iloc[-1]
        scaler = self.scalers[symbol]
        
        # Create dummy array for inverse transform
        dummy_array = np.zeros((days_ahead, recent_data.shape[1]))
        dummy_array[:, 3] = ensemble_predictions  # Close price column
        
        unscaled = scaler.inverse_transform(dummy_array)
        final_predictions = unscaled[:, 3]
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'predictions': {
                'ensemble': final_predictions.tolist(),
                'individual_models': {
                    model: self._inverse_transform_predictions(
                        preds, current_price, scaler
                    ) for model, preds in predictions.items()
                }
            },
            'prediction_dates': pd.date_range(
                start=recent_data.index[-1] + pd.Timedelta(days=1),
                periods=days_ahead
            ).tolist(),
            'confidence_intervals': self._calculate_confidence_intervals(
                predictions, final_predictions
            )
        }
    
    def _inverse_transform_predictions(self, predictions: List[float], 
                                     current_price: float, scaler) -> List[float]:
        """Inverse transform scaled predictions"""
        # Simplified inverse transform
        return [current_price * (1 + (p - 0.5) * 0.2) for p in predictions]
    
    def _calculate_confidence_intervals(self, predictions: Dict, 
                                      ensemble_preds: np.ndarray) -> Dict:
        """Calculate confidence intervals from ensemble predictions"""
        
        all_preds = np.array(list(predictions.values()))
        std_devs = np.std(all_preds, axis=0)
        
        return {
            'lower_95': (ensemble_preds - 1.96 * std_devs).tolist(),
            'upper_95': (ensemble_preds + 1.96 * std_devs).tolist(),
            'lower_68': (ensemble_preds - std_devs).tolist(),
            'upper_68': (ensemble_preds + std_devs).tolist()
        }
    
    def train_prophet_model(self, symbol: str, data: pd.DataFrame) -> Prophet:
        """Train Facebook Prophet model for comparison"""
        
        # Prepare data for Prophet
        prophet_data = pd.DataFrame({
            'ds': data.index,
            'y': data['close']
        })
        
        # Initialize and train model
        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True,
            changepoint_prior_scale=0.05
        )
        
        # Add additional regressors if available
        if 'volume' in data.columns:
            prophet_data['volume'] = data['volume']
            model.add_regressor('volume')
        
        model.fit(prophet_data)
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{symbol}_prophet_model.pkl")
        joblib.dump(model, model_path)
        
        return model
    
    def predict_prophet(self, symbol: str, days_ahead: int = 30) -> pd.DataFrame:
        """Make predictions using Prophet model"""
        
        model_path = os.path.join(self.model_dir, f"{symbol}_prophet_model.pkl")
        
        if not os.path.exists(model_path):
            raise ValueError(f"No Prophet model found for {symbol}")
        
        model = joblib.load(model_path)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=days_ahead)
        
        # If we have volume regressor, we need to provide future values
        if 'volume' in model.extra_regressors:
            # Simple assumption: use average volume
            future['volume'] = future['volume'].fillna(future['volume'].mean())
        
        # Make predictions
        forecast = model.predict(future)
        
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days_ahead) 