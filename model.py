"""
Demand Prediction Model Module
Trains a regression model to predict hourly demand per product per store.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandPredictor:
    """ML model for predicting hourly demand per product per store."""
    
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
        self.feature_cols = ['hour', 'day_of_week', 'store_id', 'product_id', 'is_weekend', 'is_peak_hour']
        
    def _prepare_features(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from order data."""
        orders = orders_df.copy()
        orders['hour'] = orders['timestamp'].dt.hour
        orders['day_of_week'] = orders['timestamp'].dt.dayofweek
        orders['is_weekend'] = (orders['day_of_week'] >= 5).astype(int)
        orders['is_peak_hour'] = orders['hour'].isin([7, 8, 9, 12, 13, 14, 18, 19, 20, 21]).astype(int)
        
        return orders
    
    def _aggregate_hourly_demand(self, orders_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate orders to hourly demand per store-product."""
        orders = self._prepare_features(orders_df)
        
        hourly_demand = orders.groupby(
            ['store_id', 'product_id', 'hour', 'day_of_week', 'is_weekend', 'is_peak_hour']
        ).agg({
            'quantity': 'sum'
        }).reset_index()
        
        hourly_demand.rename(columns={'quantity': 'actual_demand'}, inplace=True)
        
        lag_features = self._create_lag_features(hourly_demand)
        
        return lag_features
    
    def _create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create historical lag features for demand prediction."""
        df = df.sort_values(['store_id', 'product_id', 'day_of_week', 'hour'])
        
        for lag in [1, 2, 3]:
            df[f'demand_lag_{lag}'] = df.groupby(['store_id', 'product_id'])['actual_demand'].shift(lag)
        
        df['rolling_mean_3'] = df.groupby(['store_id', 'product_id'])['actual_demand'].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df['rolling_mean_7'] = df.groupby(['store_id', 'product_id'])['actual_demand'].transform(
            lambda x: x.rolling(7, min_periods=1).mean()
        )
        
        df.fillna(0, inplace=True)
        
        return df
    
    def train(self, orders_df: pd.DataFrame) -> dict:
        """Train the demand prediction model."""
        logger.info("Preparing training data...")
        
        train_data = self._aggregate_hourly_demand(orders_df)
        
        feature_cols = self.feature_cols + ['demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'rolling_mean_3', 'rolling_mean_7']
        
        X = train_data[feature_cols]
        y = train_data['actual_demand']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training on {len(X_train)} samples...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        self.is_trained = True
        
        metrics = {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Model trained. MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        return metrics
    
    def predict(self, store_id: int, product_id: int, hour: int, day_of_week: int) -> float:
        """Predict demand for a specific store-product-hour combination."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        is_weekend = 1 if day_of_week >= 5 else 0
        is_peak_hour = 1 if hour in [7, 8, 9, 12, 13, 14, 18, 19, 20, 21] else 0
        
        avg_demand = 10.0
        feature_cols = ['hour', 'day_of_week', 'store_id', 'product_id', 'is_weekend', 'is_peak_hour',
                        'demand_lag_1', 'demand_lag_2', 'demand_lag_3', 'rolling_mean_3', 'rolling_mean_7']
        
        features = pd.DataFrame([[hour, day_of_week, store_id, product_id, is_weekend, is_peak_hour,
                                  avg_demand, avg_demand, avg_demand, avg_demand, avg_demand]], 
                                columns=feature_cols)
        
        prediction = self.model.predict(features)[0]
        
        return max(0, round(prediction, 2))
    
    def predict_batch(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Predict demand for multiple store-product-hour combinations."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        df = predictions_df.copy()
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 12, 13, 14, 18, 19, 20, 21]).astype(int)
        
        X = df[self.feature_cols]
        df['predicted_demand'] = self.model.predict(X)
        df['predicted_demand'] = df['predicted_demand'].apply(lambda x: max(0, round(x, 2)))
        
        return df
    
    def get_feature_importance(self) -> dict:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        importance = self.model.feature_importances_
        feature_importance = dict(zip(self.feature_cols, importance.round(4).tolist()))
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, path: str = "demand_model.pkl"):
        """Save the trained model to disk."""
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str = "demand_model.pkl"):
        """Load a trained model from disk."""
        self.model = joblib.load(path)
        self.is_trained = True
        logger.info(f"Model loaded from {path}")


def create_prediction_input(store_ids: list, product_ids: list, hours: list = None) -> pd.DataFrame:
    """Create input dataframe for batch predictions."""
    if hours is None:
        current_hour = datetime.now().hour
        current_day = datetime.now().weekday()
        hours = list(range(24))
    
    predictions = []
    for store_id in store_ids:
        for product_id in product_ids:
            for hour in hours:
                predictions.append({
                    'store_id': store_id,
                    'product_id': product_id,
                    'hour': hour,
                    'day_of_week': datetime.now().weekday()
                })
    
    return pd.DataFrame(predictions)


if __name__ == "__main__":
    from data_generator import get_sample_data
    
    print("Loading sample data...")
    data = get_sample_data()
    
    print("\nTraining Demand Prediction Model...")
    predictor = DemandPredictor()
    metrics = predictor.train(data['orders'])
    
    print(f"\nModel Metrics:")
    print(f"  MAE: {metrics['mae']}")
    print(f"  RMSE: {metrics['rmse']}")
    print(f"  Training Samples: {metrics['training_samples']}")
    
    print("\nSample Predictions:")
    for store_id in [1, 2]:
        for product_id in [1, 2]:
            prediction = predictor.predict(store_id, product_id, 18, 3)
            print(f"  Store {store_id}, Product {product_id}, 6PM, Wed: {prediction} units")
    
    print("\nFeature Importance:")
    importance = predictor.get_feature_importance()
    for feature, score in importance.items():
        print(f"  {feature}: {score}")