# Dark Store Supply Chain Optimization AI

AI-powered system for optimizing inventory, demand forecasting, and supply chain management in quick commerce dark stores.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)
![scikit-learn](https://img.shields.io/badge/scikit-learn-1.4-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

| Module | Description |
|--------|-------------|
| **Data Simulation** | Synthetic data generation for orders, inventory, stores with time-based demand patterns |
| **Demand Prediction** | RandomForest-based hourly demand forecasting with lag features |
| **Inventory Optimization** | Stockout/overstock detection, coverage analysis |
| **Restocking Engine** | Priority-based alerts (CRITICAL/WARNING), automated restock recommendations |
| **Inter-Store Transfers** | Surplus/deficit analysis, optimal stock redistribution |
| **Expiry Management** | Perishable item tracking with clearance discount suggestions |

## 📊 Architecture

```
dark-store-optimization/
├── api.py                 # FastAPI endpoints
├── data_generator.py     # Synthetic data generation
├── model.py               # Demand prediction ML model
├── inventory.py           # Inventory optimization logic
├── transfer.py            # Inter-store transfers & expiry
├── main.py                # Application entry point
├── requirements.txt       # Python dependencies
└── README.md              # Documentation
```

## 🛠️ Tech Stack

- **Python 3.9+**
- **FastAPI** - REST API framework
- **scikit-learn** - ML model (RandomForest Regressor)
- **pandas/numpy** - Data processing
- **uvicorn** - ASGI server

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/logeshkannan19/Darkstore-supply-chain-ai.git
cd Darkstore-supply-chain-ai

# Create virtual environment (optional)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### Run the Full System

```bash
python3 main.py
```

This will:
1. Generate synthetic data (80 products, 5 stores, 10,000 orders)
2. Train the demand prediction model
3. Run analysis and print sample predictions
4. Start FastAPI server on `http://localhost:8000`

### Run API Only

```bash
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/predict-demand` | POST | Single demand prediction |
| `/predict-demand/batch` | GET | Multi-hour predictions |
| `/inventory-status` | GET | Stock status with demand comparison |
| `/restock-recommendation` | GET | Items needing restock |
| `/restock-alerts` | GET | Priority-based alerts |
| `/transfer-suggestions` | GET | Inter-store transfer recommendations |
| `/expiry-predictions` | GET | Perishable items close to expiry |
| `/clearance-recommendations` | GET | Discount suggestions for expiring items |

### Example Request

```bash
# Predict demand
curl -X POST "http://localhost:8000/predict-demand" \
  -H "Content-Type: application/json" \
  -d '{"store_id": 1, "product_id": 1, "hour": 18, "day_of_week": 3}'
```

Response:
```json
{
  "store_id": 1,
  "product_id": 1,
  "hour": 18,
  "day_of_week": 3,
  "day_name": "Wed",
  "predicted_demand": 5.54,
  "timestamp": "2026-04-01T04:50:00"
}
```

## 📈 Model Performance

The demand prediction model is evaluated using:

- **MAE (Mean Absolute Error)**: ~0.15-0.25
- **RMSE (Root Mean Squared Error)**: ~0.35-0.50

### Feature Importance

| Feature | Importance |
|---------|------------|
| hour | Time-based demand patterns |
| day_of_week | Weekly seasonality |
| product_id | Product-specific demand |
| store_id | Store location factors |
| is_peak_hour | Peak demand periods |
| demand_lag_* | Historical demand features |

## 🔧 Configuration

Modify parameters in `main.py`:

```python
generator = DataGenerator(
    n_stores=5,      # Number of dark stores
    n_products=50,  # Product catalog size
    n_orders=10000   # Training data size
)
```

## 📝 Example Output

```
============================================================
INVENTORY SUMMARY
============================================================
  total_items: 400
  stockout_risk: 68
  low_stock: 138
  healthy: 194
  overstock: 0

============================================================
SAMPLE DEMAND PREDICTIONS
============================================================
  Store 1, Product 1, 18:00 (Wed): 5.54 units
  Store 2, Product 1, 18:00 (Wed): 5.50 units
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👤 Author

**Logesh Kannan**
- GitHub: [@logeshkannan19](https://github.com/logeshkannan19)

---

⭐ Star this repo if you found it helpful!