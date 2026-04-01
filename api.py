"""
FastAPI Module for Dark Store Supply Chain Optimization
Provides API endpoints for demand prediction, inventory status, restock recommendations, and transfer suggestions.
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime
import pandas as pd
import logging

from data_generator import DataGenerator, get_sample_data
from model import DemandPredictor
from inventory import InventoryOptimizer, RestockingEngine
from transfer import TransferOptimizer, ExpiryManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Dark Store Supply Chain Optimization API",
    description="AI-powered demand prediction and inventory optimization for quick commerce",
    version="1.0.0"
)

data = None
predictor = None
optimizer = None
restock_engine = None
transfer_optimizer = None
expiry_manager = None


@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    global data, predictor, optimizer, restock_engine, transfer_optimizer, expiry_manager
    
    logger.info("Initializing Dark Store Optimization System...")
    
    logger.info("Generating synthetic data...")
    data = get_sample_data()
    
    logger.info("Training demand prediction model...")
    predictor = DemandPredictor()
    metrics = predictor.train(data['orders'])
    logger.info(f"Model trained with MAE: {metrics['mae']}, RMSE: {metrics['rmse']}")
    
    logger.info("Initializing inventory optimizer...")
    optimizer = InventoryOptimizer(data['inventory'], data['products'], predictor)
    restock_engine = RestockingEngine(optimizer)
    
    logger.info("Initializing transfer optimizer...")
    transfer_optimizer = TransferOptimizer(data['inventory'], data['stores'], data['products'])
    
    logger.info("Initializing expiry manager...")
    expiry_manager = ExpiryManager(data['inventory'], data['products'])
    
    logger.info("System ready!")


class PredictDemandRequest(BaseModel):
    store_id: int
    product_id: int
    hour: Optional[int] = None
    day_of_week: Optional[int] = None


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Dark Store Supply Chain Optimization API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": [
            "/predict-demand",
            "/inventory-status",
            "/restock-recommendation",
            "/transfer-suggestions",
            "/expiry-predictions"
        ]
    }


@app.post("/predict-demand")
async def predict_demand(request: PredictDemandRequest):
    """
    Predict demand for a specific store-product-time combination.
    
    Args:
        store_id: Store identifier
        product_id: Product identifier
        hour: Hour of day (0-23), defaults to current hour
        day_of_week: Day of week (0-6), defaults to current day
    """
    hour = request.hour if request.hour is not None else datetime.now().hour
    day_of_week = request.day_of_week if request.day_of_week is not None else datetime.now().weekday()
    
    try:
        predicted = predictor.predict(
            store_id=request.store_id,
            product_id=request.product_id,
            hour=hour,
            day_of_week=day_of_week
        )
        
        return {
            "store_id": request.store_id,
            "product_id": request.product_id,
            "hour": hour,
            "day_of_week": day_of_week,
            "day_name": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][day_of_week],
            "predicted_demand": predicted,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict-demand/batch")
async def predict_demand_batch(
    store_id: int = Query(..., description="Store ID"),
    product_id: int = Query(..., description="Product ID"),
    hours_ahead: int = Query(4, description="Number of hours to predict")
):
    """Predict demand for multiple hours ahead."""
    current_hour = datetime.now().hour
    current_day = datetime.now().weekday()
    
    predictions = []
    for h in range(hours_ahead):
        hour = (current_hour + h) % 24
        try:
            pred = predictor.predict(store_id, product_id, hour, current_day)
            predictions.append({
                "hour": hour,
                "predicted_demand": pred
            })
        except Exception as e:
            predictions.append({"hour": hour, "predicted_demand": 0, "error": str(e)})
    
    return {
        "store_id": store_id,
        "product_id": product_id,
        "predictions": predictions,
        "total_predicted_demand": sum(p["predicted_demand"] for p in predictions),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/inventory-status")
async def inventory_status(
    store_id: Optional[int] = Query(None, description="Filter by store ID"),
    status_filter: Optional[str] = Query(None, description="Filter by status: STOCKOUT_RISK, LOW_STOCK, HEALTHY, OVERSTOCK")
):
    """
    Get current inventory status with predicted demand comparison.
    
    Status values:
    - STOCKOUT_RISK: Stock is below predicted demand
    - LOW_STOCK: Stock coverage less than 2 hours
    - HEALTHY: Stock is adequate
    - OVERSTOCK: Stock excessive (coverage > 10 hours)
    """
    try:
        stock_status = optimizer.analyze_stock_status()
        
        if store_id:
            stock_status = stock_status[stock_status['store_id'] == store_id]
        
        if status_filter:
            stock_status = stock_status[stock_status['status'] == status_filter]
        
        stock_status = stock_status.merge(
            data['products'][['product_id', 'product_name', 'category']],
            on='product_id'
        )
        
        return {
            "total_items": len(stock_status),
            "summary": {
                "stockout_risk": int((stock_status['status'] == 'STOCKOUT_RISK').sum()),
                "low_stock": int((stock_status['status'] == 'LOW_STOCK').sum()),
                "healthy": int((stock_status['status'] == 'HEALTHY').sum()),
                "overstock": int((stock_status['status'] == 'OVERSTOCK').sum())
            },
            "items": stock_status.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Inventory status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory-status/summary")
async def inventory_summary():
    """Get inventory summary statistics."""
    summary = optimizer.get_inventory_summary()
    return summary


@app.get("/restock-recommendation")
async def restock_recommendation(
    store_id: Optional[int] = Query(None, description="Filter by store ID"),
    threshold: float = Query(1.5, description="Threshold multiplier for restock calculation")
):
    """
    Get restock recommendations for items at risk.
    
    Returns items where current stock is less than predicted demand,
    with recommended order quantities.
    """
    try:
        recommendations = optimizer.get_restock_recommendations(threshold_multiplier=threshold)
        
        if store_id:
            recommendations = recommendations[recommendations['store_id'] == store_id]
        
        return {
            "total_recommendations": len(recommendations),
            "perishable_items": int(recommendations['is_perishable'].sum()),
            "recommendations": recommendations.to_dict('records'),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Restock recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/restock-alerts")
async def restock_alerts(store_id: Optional[int] = Query(None)):
    """Get active restock alerts with priority levels."""
    alerts = restock_engine.check_restock_needs(store_id)
    
    return {
        "total_alerts": len(alerts),
        "critical": len([a for a in alerts if a['priority'] == 'CRITICAL']),
        "warning": len([a for a in alerts if a['priority'] == 'WARNING']),
        "alerts": alerts,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/restock-order")
async def create_restock_order(
    store_id: int = Query(..., description="Store ID"),
    product_id: int = Query(..., description="Product ID")
):
    """Generate a restock order for a specific item."""
    order = restock_engine.generate_restock_order(store_id, product_id)
    
    if 'error' in order:
        raise HTTPException(status_code=404, detail=order['error'])
    
    return {
        "order": order,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/transfer-suggestions")
async def transfer_suggestions(
    product_id: Optional[int] = Query(None, description="Filter by product ID"),
    max_transfers: int = Query(20, description="Maximum number of transfers to suggest"),
    optimize_distance: bool = Query(False, description="Optimize for distance (within 10km)")
):
    """
    Get inter-store transfer suggestions to balance inventory.
    
    Identifies stores with surplus stock and suggests transfers
    to stores with deficit.
    """
    try:
        if optimize_distance:
            transfers = transfer_optimizer.optimize_transfers_with_distance(max_distance_km=10)
        else:
            transfers = transfer_optimizer.suggest_transfers(product_id, max_transfers)
        
        return {
            "total_transfers": len(transfers),
            "total_units": sum(t['transfer_quantity'] for t in transfers),
            "transfers": transfers,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Transfer suggestions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/transfer-summary")
async def transfer_summary():
    """Get summary of all transfer recommendations."""
    summary = transfer_optimizer.get_transfer_summary()
    return summary


@app.get("/surplus-deficit-analysis")
async def surplus_deficit_analysis(product_id: Optional[int] = Query(None)):
    """Get detailed surplus/deficit analysis by store."""
    analysis = transfer_optimizer.calculate_surplus_deficit(product_id)
    return {
        "items_analyzed": len(analysis),
        "surplus_stores": int((analysis['status'] == 'SURPLUS').sum()),
        "deficit_stores": int((analysis['status'] == 'DEFICIT').sum()),
        "balanced_stores": int((analysis['status'] == 'BALANCED').sum()),
        "analysis": analysis.to_dict('records')
    }


@app.get("/expiry-predictions")
async def expiry_predictions(days_threshold: int = Query(3, description="Days threshold for expiry alerts")):
    """Get expiry predictions for perishable items."""
    expiry_data = expiry_manager.get_expiry_predictions()
    
    return {
        "total_perishable_items": len(expiry_data),
        "items": expiry_data.to_dict('records') if not expiry_data.empty else [],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/clearance-recommendations")
async def clearance_recommendations(days_threshold: int = Query(3)):
    """Get clearance/discount recommendations for items close to expiry."""
    clearance = expiry_manager.get_clearance_recommendations(days_threshold)
    
    if clearance.empty:
        return {
            "message": "No items require immediate clearance",
            "recommendations": []
        }
    
    return {
        "total_items": len(clearance),
        "total_units": int(clearance['stock_level'].sum()),
        "total_value_at_discount": round(clearance['total_value_at_discount'].sum(), 2),
        "recommendations": clearance.to_dict('records'),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/expiry-summary")
async def expiry_summary():
    """Get summary of expiry situation across all stores."""
    summary = expiry_manager.get_expiry_summary()
    return summary


@app.get("/model/metrics")
async def model_metrics():
    """Get model training metrics."""
    feature_importance = predictor.get_feature_importance()
    
    return {
        "feature_importance": feature_importance,
        "is_trained": predictor.is_trained
    }


@app.get("/stores")
async def get_stores():
    """Get all store information."""
    return {
        "stores": data['stores'].to_dict('records'),
        "total": len(data['stores'])
    }


@app.get("/products")
async def get_products(
    category: Optional[str] = Query(None, description="Filter by category"),
    perishable_only: bool = Query(False, description="Filter only perishable items")
):
    """Get product catalog."""
    products = data['products'].copy()
    
    if category:
        products = products[products['category'] == category]
    
    if perishable_only:
        products = products[products['is_perishable'] == True]
    
    return {
        "total_products": len(products),
        "categories": products['category'].unique().tolist(),
        "products": products.to_dict('records')
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)