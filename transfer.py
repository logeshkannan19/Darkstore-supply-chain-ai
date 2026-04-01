"""
Inter-Store Transfer Module
Identifies surplus stores and deficit stores, suggests optimal stock transfers.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import List, Dict, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TransferOptimizer:
    """Optimizes stock transfers between stores to balance inventory."""
    
    def __init__(self, inventory_df: pd.DataFrame, stores_df: pd.DataFrame, products_df: pd.DataFrame):
        self.inventory = inventory_df.copy()
        self.stores = stores_df.copy()
        self.products = products_df.copy()
        
    def calculate_surplus_deficit(self, product_id: int = None, threshold: float = 1.5) -> pd.DataFrame:
        """Calculate surplus and deficit for each store-product combination."""
        df = self.inventory.copy()
        
        if product_id:
            df = df[df['product_id'] == product_id]
        
        df['avg_stock'] = df.groupby('product_id')['stock_level'].transform('mean')
        df['stock_ratio'] = df['stock_level'] / df['avg_stock']
        
        df['status'] = 'BALANCED'
        df.loc[df['stock_ratio'] > threshold, 'status'] = 'SURPLUS'
        df.loc[df['stock_ratio'] < (1/threshold), 'status'] = 'DEFICIT'
        
        df = df.merge(self.stores[['store_id', 'store_name', 'location']], on='store_id')
        df = df.merge(self.products[['product_id', 'product_name', 'category']], on='product_id')
        
        return df[['store_id', 'store_name', 'location', 'product_id', 'product_name', 
                   'category', 'stock_level', 'avg_stock', 'stock_ratio', 'status']]
    
    def suggest_transfers(self, product_id: int = None, max_transfers: int = 20) -> List[Dict]:
        """Suggest optimal transfers between stores."""
        stock_analysis = self.calculate_surplus_deficit(product_id)
        
        transfers = []
        
        for prod_id in stock_analysis['product_id'].unique():
            product_data = stock_analysis[stock_analysis['product_id'] == prod_id]
            
            surplus_stores = product_data[product_data['status'] == 'SURPLUS'].sort_values('stock_ratio', ascending=False)
            deficit_stores = product_data[product_data['status'] == 'DEFICIT'].sort_values('stock_ratio')
            
            if surplus_stores.empty or deficit_stores.empty:
                continue
            
            for _, deficit in deficit_stores.iterrows():
                if len(transfers) >= max_transfers:
                    break
                    
                for _, surplus in surplus_stores.iterrows():
                    if surplus['stock_level'] <= deficit['avg_stock']:
                        continue
                    
                    transfer_qty = min(
                        int((surplus['stock_level'] - surplus['avg_stock']) / 2),
                        int(deficit['avg_stock'] - deficit['stock_level'])
                    )
                    
                    if transfer_qty <= 0:
                        continue
                    
                    transfer = {
                        'product_id': prod_id,
                        'product_name': deficit['product_name'],
                        'from_store_id': surplus['store_id'],
                        'from_store_name': surplus['store_name'],
                        'from_location': surplus['location'],
                        'to_store_id': deficit['store_id'],
                        'to_store_name': deficit['store_name'],
                        'to_location': deficit['location'],
                        'transfer_quantity': transfer_qty,
                        'reason': f"Balance stock: {surplus['store_name']} has surplus, {deficit['store_name']} has deficit"
                    }
                    transfers.append(transfer)
                    
                    surplus_stores.loc[surplus_stores['store_id'] == surplus['store_id'], 'stock_level'] -= transfer_qty
                    deficit_stores.loc[deficit_stores['store_id'] == deficit['store_id'], 'stock_level'] += transfer_qty
                    
                    break
                
                if len(transfers) >= max_transfers:
                    break
        
        logger.info(f"Suggested {len(transfers)} stock transfers")
        return transfers
    
    def get_transfer_summary(self) -> Dict:
        """Get summary of all transfer recommendations."""
        transfers = self.suggest_transfers()
        
        if not transfers:
            return {
                'total_transfers': 0,
                'total_units': 0,
                'products_affected': 0,
                'stores_involved': 0,
                'message': 'No transfers needed - inventory is balanced'
            }
        
        unique_products = len(set(t['product_id'] for t in transfers))
        unique_stores = len(set(t['from_store_id'] for t in transfers) | set(t['to_store_id'] for t in transfers))
        total_units = sum(t['transfer_quantity'] for t in transfers)
        
        return {
            'total_transfers': len(transfers),
            'total_units': total_units,
            'products_affected': unique_products,
            'stores_involved': unique_stores,
            'transfers': transfers
        }
    
    def calculate_transfer_distance(self, from_store_id: int, to_store_id: int) -> float:
        """Calculate approximate distance between two stores in km."""
        from_store = self.stores[self.stores['store_id'] == from_store_id].iloc[0]
        to_store = self.stores[self.stores['store_id'] == to_store_id].iloc[0]
        
        lat1, lon1 = from_store['latitude'], from_store['longitude']
        lat2, lon2 = to_store['latitude'], to_store['longitude']
        
        return np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111
    
    def optimize_transfers_with_distance(self, max_distance_km: float = 10) -> List[Dict]:
        """Suggest transfers considering distance between stores."""
        all_transfers = self.suggest_transfers()
        
        optimized = []
        for transfer in all_transfers:
            distance = self.calculate_transfer_distance(
                transfer['from_store_id'], 
                transfer['to_store_id']
            )
            
            if distance <= max_distance_km:
                transfer['distance_km'] = round(distance, 2)
                optimized.append(transfer)
        
        logger.info(f"Optimized {len(optimized)} transfers within {max_distance_km}km")
        return optimized


class ExpiryManager:
    """Manages perishable items and predicts expiry for clearance suggestions."""
    
    def __init__(self, inventory_df: pd.DataFrame, products_df: pd.DataFrame):
        self.inventory = inventory_df.copy()
        self.products = products_df.copy()
        
    def get_expiry_predictions(self) -> pd.DataFrame:
        """Get all items with expiry dates and days until expiry."""
        inventory_with_expiry = self.inventory[self.inventory['expiry_date'].notna()].copy()
        
        if inventory_with_expiry.empty:
            return pd.DataFrame()
        
        inventory_with_expiry['days_until_expiry'] = (
            pd.to_datetime(inventory_with_expiry['expiry_date']) - pd.Timestamp.now()
        ).dt.days
        
        inventory_with_expiry = inventory_with_expiry.merge(
            self.products[['product_id', 'product_name', 'category', 'base_price', 'is_perishable']],
            on='product_id'
        )
        
        return inventory_with_expiry[['store_id', 'product_id', 'product_name', 'category',
                                     'stock_level', 'expiry_date', 'days_until_expiry', 'base_price']]
    
    def get_clearance_recommendations(self, days_threshold: int = 3) -> pd.DataFrame:
        """Recommend items for clearance/discount based on expiry."""
        expiry_data = self.get_expiry_predictions()
        
        if expiry_data.empty:
            return pd.DataFrame()
        
        clearance = expiry_data[expiry_data['days_until_expiry'] <= days_threshold].copy()
        
        if not clearance.empty:
            clearance['discount_percentage'] = 0
            clearance.loc[clearance['days_until_expiry'] <= 1, 'discount_percentage'] = 50
            clearance.loc[clearance['days_until_expiry'] == 2, 'discount_percentage'] = 30
            clearance.loc[clearance['days_until_expiry'] == 3, 'discount_percentage'] = 20
            
            clearance['discounted_price'] = clearance['base_price'] * (1 - clearance['discount_percentage'] / 100)
            clearance['total_value_at_discount'] = clearance['stock_level'] * clearance['discounted_price']
        
        return clearance
    
    def get_expiry_summary(self) -> Dict:
        """Get summary of expiry situation."""
        expiry_data = self.get_expiry_predictions()
        
        if expiry_data.empty:
            return {
                'total_perishable_items': 0,
                'items_expiring_today': 0,
                'items_expiring_3_days': 0,
                'total_units_at_risk': 0,
                'clearance_recommendations': []
            }
        
        return {
            'total_perishable_items': len(expiry_data),
            'items_expiring_today': len(expiry_data[expiry_data['days_until_expiry'] <= 0]),
            'items_expiring_3_days': len(expiry_data[expiry_data['days_until_expiry'] <= 3]),
            'total_units_at_risk': int(expiry_data[expiry_data['days_until_expiry'] <= 3]['stock_level'].sum()),
            'clearance_recommendations': self.get_clearance_recommendations().to_dict('records')
        }


if __name__ == "__main__":
    from data_generator import get_sample_data
    
    print("Loading sample data...")
    data = get_sample_data()
    
    print("\n--- Inter-Store Transfer Analysis ---")
    transfer_optimizer = TransferOptimizer(data['inventory'], data['stores'], data['products'])
    
    stock_analysis = transfer_optimizer.calculate_surplus_deficit()
    print("\nStock Status by Store:")
    print(stock_analysis.groupby(['store_name', 'status']).size().unstack(fill_value=0))
    
    print("\n--- Transfer Suggestions ---")
    transfers = transfer_optimizer.suggest_transfers()
    print(f"Total transfers suggested: {len(transfers)}")
    if transfers:
        print(transfers[0])
    
    print("\n--- Expiry Management ---")
    expiry_manager = ExpiryManager(data['inventory'], data['products'])
    expiry_summary = expiry_manager.get_expiry_summary()
    print(f"Perishable items: {expiry_summary['total_perishable_items']}")
    print(f"Expiring within 3 days: {expiry_summary['items_expiring_3_days']}")
    print(f"Units at risk: {expiry_summary['total_units_at_risk']}")