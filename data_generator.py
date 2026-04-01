"""
Data Generator Module for Dark Store Supply Chain Optimization
Generates synthetic datasets for orders, inventory, and store metadata.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataGenerator:
    """Generates synthetic data for quick commerce dark store simulation."""
    
    def __init__(self, n_stores: int = 5, n_products: int = 50, n_orders: int = 10000):
        self.n_stores = n_stores
        self.n_products = n_products
        self.n_orders = n_orders
        self.products = self._generate_products()
        self.stores = self._generate_stores()
        self._n_products_actual = len(self.products)
        
    def _generate_products(self) -> pd.DataFrame:
        """Generate product catalog with categories and expiry info."""
        categories = ['Groceries', 'Dairy', 'Beverages', 'Snacks', 'Bakery', 'Frozen', 'Household', 'Personal Care']
        product_types = {
            'Groceries': ['Rice', 'Sugar', 'Oil', 'Pulses', 'Spices'],
            'Dairy': ['Milk', 'Yogurt', 'Cheese', 'Butter', 'Cream'],
            'Beverages': ['Water', 'Juice', 'Soda', 'Tea', 'Coffee'],
            'Snacks': ['Chips', 'Biscuits', 'Nuts', 'Chocolate', 'Candy'],
            'Bakery': ['Bread', 'Bun', 'Cake', 'Pastry', 'Croissant'],
            'Frozen': ['Ice Cream', 'Frozen Veg', 'Frozen Meat', 'Pizza', 'Frozen Snacks'],
            'Household': ['Detergent', 'Cleaner', 'Tissue', 'Batteries', 'Bulbs'],
            'Personal Care': ['Shampoo', 'Soap', 'Toothpaste', 'Deodorant', 'Lotion']
        }
        
        products = []
        product_id = 1
        
        for category, items in product_types.items():
            for item in items:
                for variant in range(1, 3):
                    is_perishable = category in ['Dairy', 'Bakery', 'Frozen']
                    expiry_days = random.randint(1, 7) if is_perishable else random.randint(30, 180)
                    products.append({
                        'product_id': product_id,
                        'product_name': f"{item} {variant}L" if category == 'Dairy' else f"{item}",
                        'category': category,
                        'is_perishable': is_perishable,
                        'base_price': round(random.uniform(10, 500), 2),
                        'expiry_days': expiry_days
                    })
                    product_id += 1
        
        df = pd.DataFrame(products)
        logger.info(f"Generated {len(df)} products across {len(categories)} categories")
        return df
    
    def _generate_stores(self) -> pd.DataFrame:
        """Generate dark store metadata."""
        locations = [
            ('Downtown', 40.7128, -74.0060),
            ('Midtown', 40.7549, -73.9840),
            ('Uptown', 40.7831, -73.9712),
            ('Brooklyn', 40.6782, -73.9442),
            ('Queens', 40.7282, -73.7949),
            ('Bronx', 40.8448, -73.8648),
            ('Jersey City', 40.7178, -74.0431),
            ('Hoboken', 40.7440, -74.0324)
        ]
        
        stores = []
        for i in range(min(self.n_stores, len(locations))):
            name, lat, lng = locations[i]
            stores.append({
                'store_id': i + 1,
                'store_name': f"Dark Store {name}",
                'location': name,
                'latitude': lat + np.random.uniform(-0.02, 0.02),
                'longitude': lng + np.random.uniform(-0.02, 0.02),
                'capacity': random.randint(500, 2000),
                'zone_type': random.choice(['Residential', 'Commercial', 'Mixed'])
            })
        
        df = pd.DataFrame(stores)
        logger.info(f"Generated {len(df)} dark stores")
        return df
    
    def _generate_hourly_demand_pattern(self, hour: int, day_of_week: int) -> float:
        """Generate time-based demand patterns (peak hours, weekends)."""
        base_demand = 1.0
        
        if hour in [7, 8, 9]:
            base_demand *= 2.5
        elif hour in [12, 13, 14]:
            base_demand *= 1.8
        elif hour in [18, 19, 20, 21]:
            base_demand *= 3.0
        elif hour in [0, 1, 2, 3, 4, 5]:
            base_demand *= 0.3
        
        if day_of_week in [5, 6]:
            base_demand *= 1.3
        
        return base_demand
    
    def generate_orders(self) -> pd.DataFrame:
        """Generate order data with time-based demand patterns."""
        orders = []
        start_date = datetime.now() - timedelta(days=30)
        
        for _ in range(self.n_orders):
            hour_probs = [0.02, 0.01, 0.01, 0.01, 0.02, 0.05, 0.10, 0.15, 
                   0.08, 0.05, 0.05, 0.06, 0.08, 0.06, 0.05, 0.05,
                   0.06, 0.08, 0.10, 0.12, 0.10, 0.08, 0.05, 0.03]
            hour_probs = [p / sum(hour_probs) for p in hour_probs]
            hour = int(np.random.choice(range(24), p=hour_probs))
            day_of_week = (start_date + timedelta(days=random.randint(0, 30))).weekday()
            
            demand_multiplier = self._generate_hourly_demand_pattern(hour, day_of_week)
            
            product_weights = np.random.dirichlet(np.ones(len(self.products)) * 0.1)
            product_id = int(np.random.choice(self.products['product_id'].values, p=product_weights))
            
            base_quantity = np.random.choice([1, 1, 1, 2, 2, 3], p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03])
            quantity = int(base_quantity * demand_multiplier)
            
            timestamp = start_date + timedelta(
                days=random.randint(0, 30),
                hours=hour,
                minutes=random.randint(0, 59)
            )
            
            orders.append({
                'order_id': len(orders) + 1,
                'store_id': random.randint(1, self.n_stores),
                'timestamp': timestamp,
                'product_id': product_id,
                'quantity': max(1, quantity)
            })
        
        df = pd.DataFrame(orders).sort_values('timestamp').reset_index(drop=True)
        logger.info(f"Generated {len(df)} orders")
        return df
    
    def generate_inventory(self) -> pd.DataFrame:
        """Generate initial inventory levels for each store-product combination."""
        inventory = []
        
        for store_id in range(1, self.n_stores + 1):
            store_capacity = self.stores[self.stores['store_id'] == store_id]['capacity'].values[0]
            
            for product_id in self.products['product_id']:
                is_perishable = self.products[self.products['product_id'] == product_id]['is_perishable'].values[0]
                
                if is_perishable:
                    stock_level = random.randint(5, 30)
                else:
                    stock_level = random.randint(20, 100)
                
                expiry_date = None
                if is_perishable:
                    days_until_expiry = random.randint(1, 7)
                    expiry_date = datetime.now() + timedelta(days=days_until_expiry)
                
                inventory.append({
                    'store_id': store_id,
                    'product_id': product_id,
                    'stock_level': stock_level,
                    'reorder_point': random.randint(10, 25),
                    'reorder_quantity': random.randint(20, 50),
                    'expiry_date': expiry_date,
                    'last_restocked': datetime.now() - timedelta(days=random.randint(0, 3))
                })
        
        df = pd.DataFrame(inventory)
        logger.info(f"Generated inventory for {len(df)} store-product combinations")
        return df
    
    def generate_all(self) -> dict:
        """Generate all datasets and return as dictionary."""
        return {
            'products': self.products,
            'stores': self.stores,
            'orders': self.generate_orders(),
            'inventory': self.generate_inventory()
        }


def get_sample_data() -> dict:
    """Utility function to get sample data for testing."""
    generator = DataGenerator(n_stores=5, n_products=50, n_orders=5000)
    return generator.generate_all()


if __name__ == "__main__":
    data = get_sample_data()
    print("Sample Data Generated:")
    print(f"Products: {len(data['products'])}")
    print(f"Stores: {len(data['stores'])}")
    print(f"Orders: {len(data['orders'])}")
    print(f"Inventory: {len(data['inventory'])}")
    print("\nProducts Sample:")
    print(data['products'].head())
    print("\nOrders Sample:")
    print(data['orders'].head())