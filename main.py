"""
Main entry point for Dark Store Supply Chain Optimization System
Run this to train models, generate data, and start the API server.
"""

import logging
import sys
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('dark_store_optimization.log')
    ]
)
logger = logging.getLogger(__name__)


def main():
    """Main function to run the complete system."""
    logger.info("=" * 60)
    logger.info("Dark Store Supply Chain Optimization System")
    logger.info("=" * 60)
    
    logger.info("\n[1/5] Importing modules...")
    from data_generator import DataGenerator, get_sample_data
    from model import DemandPredictor
    from inventory import InventoryOptimizer, RestockingEngine
    from transfer import TransferOptimizer, ExpiryManager
    
    logger.info("\n[2/5] Generating synthetic data...")
    generator = DataGenerator(n_stores=5, n_products=50, n_orders=10000)
    data = generator.generate_all()
    
    logger.info(f"  - Products: {len(data['products'])}")
    logger.info(f"  - Stores: {len(data['stores'])}")
    logger.info(f"  - Orders: {len(data['orders'])}")
    logger.info(f"  - Inventory records: {len(data['inventory'])}")
    
    logger.info("\n[3/5] Training Demand Prediction Model...")
    predictor = DemandPredictor()
    metrics = predictor.train(data['orders'])
    
    logger.info(f"  - Training samples: {metrics['training_samples']}")
    logger.info(f"  - Test samples: {metrics['test_samples']}")
    logger.info(f"  - MAE: {metrics['mae']}")
    logger.info(f"  - RMSE: {metrics['rmse']}")
    
    logger.info("\n[4/5] Initializing Optimizers...")
    optimizer = InventoryOptimizer(data['inventory'], data['products'], predictor)
    restock_engine = RestockingEngine(optimizer)
    transfer_optimizer = TransferOptimizer(data['inventory'], data['stores'], data['products'])
    expiry_manager = ExpiryManager(data['inventory'], data['products'])
    
    logger.info("\n[5/5] Running Sample Analysis...")
    
    print("\n" + "=" * 60)
    print("INVENTORY SUMMARY")
    print("=" * 60)
    summary = optimizer.get_inventory_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("SAMPLE DEMAND PREDICTIONS")
    print("=" * 60)
    for store_id in [1, 2, 3]:
        for hour in [8, 12, 18, 21]:
            pred = predictor.predict(store_id, 1, hour, 3)
            print(f"  Store {store_id}, Product 1, {hour}:00 (Wed): {pred} units")
    
    print("\n" + "=" * 60)
    print("RESTOCK RECOMMENDATIONS")
    print("=" * 60)
    restock = optimizer.get_restock_recommendations()
    print(f"  Items needing restock: {len(restock)}")
    if len(restock) > 0:
        print(restock[['store_id', 'product_id', 'current_stock', 'predicted_demand_4h', 'recommended_order_quantity']].head())
    
    print("\n" + "=" * 60)
    print("TRANSFER SUGGESTIONS")
    print("=" * 60)
    transfers = transfer_optimizer.suggest_transfers(max_transfers=10)
    print(f"  Suggested transfers: {len(transfers)}")
    if transfers:
        for t in transfers[:3]:
            print(f"    {t['from_store_name']} -> {t['to_store_name']}: {t['transfer_quantity']} x {t['product_name']}")
    
    print("\n" + "=" * 60)
    print("EXPIRY PREDICTIONS")
    print("=" * 60)
    expiry_summary = expiry_manager.get_expiry_summary()
    print(f"  Perishable items: {expiry_summary['total_perishable_items']}")
    print(f"  Expiring within 3 days: {expiry_summary['items_expiring_3_days']}")
    print(f"  Units at risk: {expiry_summary['total_units_at_risk']}")
    
    print("\n" + "=" * 60)
    print("MODEL FEATURE IMPORTANCE")
    print("=" * 60)
    importance = predictor.get_feature_importance()
    for feature, score in importance.items():
        print(f"  {feature}: {score:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Starting FastAPI Server on http://localhost:8000")
    logger.info("=" * 60 + "\n")
    
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()