"""
Inventory Optimization Module
Compares predicted demand vs current stock, identifies stockout/overstock risks,
and suggests restock quantities.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InventoryOptimizer:
    """Handles inventory optimization, stock analysis, and restock recommendations."""
    
    def __init__(self, inventory_df: pd.DataFrame, products_df: pd.DataFrame, predictor=None):
        self.inventory = inventory_df.copy()
        self.products = products_df.copy()
        self.predictor = predictor
        self.current_hour = datetime.now().hour
        self.current_day = datetime.now().weekday()
        
    def analyze_stock_status(self, hours_ahead: int = 4) -> pd.DataFrame:
        """Analyze current stock status vs predicted demand."""
        results = []
        
        for _, row in self.inventory.iterrows():
            store_id = row['store_id']
            product_id = row['product_id']
            current_stock = row['stock_level']
            
            predicted_demand = 0
            for hour in range(self.current_hour, self.current_hour + hours_ahead):
                if self.predictor:
                    hour_mod = hour % 24
                    pred = self.predictor.predict(store_id, product_id, hour_mod, self.current_day)
                    predicted_demand += pred
                else:
                    predicted_demand += np.random.uniform(5, 15)
            
            stock_coverage = current_stock / predicted_demand if predicted_demand > 0 else float('inf')
            
            if current_stock < predicted_demand:
                status = 'STOCKOUT_RISK'
            elif stock_coverage < 2:
                status = 'LOW_STOCK'
            elif stock_coverage > 10:
                status = 'OVERSTOCK'
            else:
                status = 'HEALTHY'
            
            results.append({
                'store_id': store_id,
                'product_id': product_id,
                'current_stock': current_stock,
                'predicted_demand_4h': round(predicted_demand, 2),
                'stock_coverage_hours': round(stock_coverage, 2),
                'status': status,
                'reorder_needed': current_stock < predicted_demand
            })
        
        df = pd.DataFrame(results)
        logger.info(f"Analyzed stock status for {len(df)} store-product combinations")
        return df
    
    def get_restock_recommendations(self, threshold_multiplier: float = 1.5) -> pd.DataFrame:
        """Generate restock recommendations for items at risk."""
        stock_status = self.analyze_stock_status()
        
        recommendations = stock_status[stock_status['reorder_needed']].copy()
        
        recommendations['recommended_order_quantity'] = (
            recommendations['predicted_demand_4h'] * threshold_multiplier
        ).astype(int)
        
        recommendations = recommendations.merge(
            self.products[['product_id', 'product_name', 'category', 'is_perishable']],
            on='product_id'
        )
        
        recommendations = recommendations.merge(
            self.inventory[['store_id', 'product_id', 'reorder_point', 'reorder_quantity']],
            on=['store_id', 'product_id']
        )
        
        logger.info(f"Generated {len(recommendations)} restock recommendations")
        return recommendations[['store_id', 'product_id', 'product_name', 'category', 
                               'current_stock', 'predicted_demand_4h', 
                               'recommended_order_quantity', 'is_perishable']]
    
    def get_overstock_items(self) -> pd.DataFrame:
        """Identify overstocked items that could be transferred or discounted."""
        stock_status = self.analyze_stock_status()
        
        overstock = stock_status[stock_status['status'] == 'OVERSTOCK'].copy()
        
        overstock = overstock.merge(
            self.products[['product_id', 'product_name', 'category', 'is_perishable', 'expiry_days']],
            on='product_id'
        )
        
        logger.info(f"Found {len(overstock)} overstock items")
        return overstock
    
    def get_expiring_soon(self, days_threshold: int = 2) -> pd.DataFrame:
        """Identify items close to expiry that need clearance."""
        inventory_with_expiry = self.inventory[self.inventory['expiry_date'].notna()].copy()
        
        if inventory_with_expiry.empty:
            return pd.DataFrame()
        
        inventory_with_expiry['days_until_expiry'] = (
            pd.to_datetime(inventory_with_expiry['expiry_date']) - pd.Timestamp.now()
        ).dt.days
        
        expiring = inventory_with_expiry[
            inventory_with_expiry['days_until_expiry'] <= days_threshold
        ].copy()
        
        if not expiring.empty:
            expiring = expiring.merge(
                self.products[['product_id', 'product_name', 'category', 'base_price']],
                on='product_id'
            )
            
            expiring['discount_recommendation'] = '30%'
            expiring.loc[expiring['days_until_expiry'] <= 1, 'discount_recommendation'] = '50%'
        
        logger.info(f"Found {len(expiring)} items expiring soon")
        return expiring
    
    def get_inventory_summary(self) -> dict:
        """Get summary statistics of inventory status."""
        stock_status = self.analyze_stock_status()
        
        summary = {
            'total_items': len(stock_status),
            'stockout_risk': len(stock_status[stock_status['status'] == 'STOCKOUT_RISK']),
            'low_stock': len(stock_status[stock_status['status'] == 'LOW_STOCK']),
            'healthy': len(stock_status[stock_status['status'] == 'HEALTHY']),
            'overstock': len(stock_status[stock_status['status'] == 'OVERSTOCK']),
            'total_units_in_stock': int(stock_status['current_stock'].sum()),
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def update_inventory(self, store_id: int, product_id: int, quantity_change: int):
        """Update inventory levels after restocking or sales."""
        mask = (self.inventory['store_id'] == store_id) & (self.inventory['product_id'] == product_id)
        
        if mask.any():
            self.inventory.loc[mask, 'stock_level'] = (
                self.inventory.loc[mask, 'stock_level'] + quantity_change
            ).clip(lower=0)
            logger.info(f"Updated inventory: Store {store_id}, Product {product_id}, Change: {quantity_change}")
        else:
            logger.warning(f"Inventory record not found for Store {store_id}, Product {product_id}")


class RestockingEngine:
    """Engine for triggering restock alerts and recommendations."""
    
    def __init__(self, optimizer: InventoryOptimizer):
        self.optimizer = optimizer
        self.alert_thresholds = {
            'critical': 0.5,
            'warning': 1.0,
            'safe': 2.0
        }
        
    def check_restock_needs(self, store_id: int = None) -> list:
        """Check which items need restocking."""
        stock_status = self.optimizer.analyze_stock_status()
        
        if store_id:
            stock_status = stock_status[stock_status['store_id'] == store_id]
        
        alerts = []
        
        for _, row in stock_status.iterrows():
            if row['status'] == 'STOCKOUT_RISK':
                priority = 'CRITICAL'
            elif row['status'] == 'LOW_STOCK':
                priority = 'WARNING'
            else:
                continue
            
            alerts.append({
                'store_id': row['store_id'],
                'product_id': row['product_id'],
                'current_stock': row['current_stock'],
                'predicted_demand': row['predicted_demand_4h'],
                'priority': priority,
                'action': 'RESTOCK_IMMEDIATE' if priority == 'CRITICAL' else 'RESTOCK_SOON'
            })
        
        return alerts
    
    def generate_restock_order(self, store_id: int, product_id: int) -> dict:
        """Generate a restock order for a specific item."""
        stock_status = self.optimizer.analyze_stock_status()
        item = stock_status[
            (stock_status['store_id'] == store_id) & 
            (stock_status['product_id'] == product_id)
        ]
        
        if item.empty:
            return {'error': 'Item not found'}
        
        item = item.iloc[0]
        predicted_demand = item['predicted_demand_4h']
        current_stock = item['current_stock']
        
        reorder_qty = max(0, int(predicted_demand * 2) - current_stock)
        
        return {
            'store_id': store_id,
            'product_id': product_id,
            'current_stock': current_stock,
            'predicted_demand_4h': predicted_demand,
            'recommended_reorder_qty': reorder_qty,
            'order_date': datetime.now().isoformat(),
            'expected_delivery': (datetime.now() + timedelta(hours=2)).isoformat()
        }


if __name__ == "__main__":
    from data_generator import get_sample_data
    from model import DemandPredictor
    
    print("Loading data and training model...")
    data = get_sample_data()
    
    predictor = DemandPredictor()
    predictor.train(data['orders'])
    
    print("\nInitializing Inventory Optimizer...")
    optimizer = InventoryOptimizer(data['inventory'], data['products'], predictor)
    
    print("\n--- Inventory Summary ---")
    summary = optimizer.get_inventory_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n--- Stock Status Analysis ---")
    stock_status = optimizer.analyze_stock_status()
    print(stock_status['status'].value_counts())
    
    print("\n--- Restock Recommendations ---")
    restock = optimizer.get_restock_recommendations()
    print(f"Items needing restock: {len(restock)}")
    if len(restock) > 0:
        print(restock.head(10))
    
    print("\n--- Expiring Soon ---")
    expiring = optimizer.get_expiring_soon()
    print(f"Items expiring within 2 days: {len(expiring)}")
    if len(expiring) > 0:
        print(expiring[['store_id', 'product_id', 'days_until_expiry', 'discount_recommendation']].head())