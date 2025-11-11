# ML Models Directory

## Purpose

This directory stores trained machine learning models for demand forecasting.

## How Models Are Created

Models are **automatically generated** when you first run a forecast:

```bash
# This will train models and save them here
curl -X POST "http://localhost:8000/api/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{"product_id": 1, "forecast_horizon": 90}'
```

## What Gets Saved

For each product, the following models are trained and saved:

### 1. Prophet Model (`prophet_model.pkl`)
- Facebook's time series forecasting model
- Captures trend, seasonality, and holidays
- File size: ~500KB - 2MB

### 2. LSTM Model (`lstm_model.h5`)
- Deep learning neural network
- 4-layer architecture (128-64-32-16)
- File size: ~5-10MB

### 3. Random Forest (`rf_model.pkl`)
- Ensemble of 200 decision trees
- Uses 20+ engineered features
- File size: ~10-20MB

### 4. Scaler (`scaler.pkl`)
- StandardScaler for feature normalization
- Required for LSTM predictions
- File size: ~10KB

## Directory Structure

```
ml_models/
└── forecasting/
    ├── 1/                      # Product ID 1
    │   ├── prophet_model.pkl
    │   ├── lstm_model.h5
    │   ├── rf_model.pkl
    │   ├── scaler.pkl
    │   └── model_info.json
    ├── 2/                      # Product ID 2
    │   ├── prophet_model.pkl
    │   └── ...
    └── ...
```

## Model Training Time

First forecast (with training):
- **Prophet**: 5-10 seconds
- **LSTM**: 30-60 seconds
- **Random Forest**: 10-20 seconds
- **Total**: ~1-2 minutes

Subsequent forecasts (using saved models):
- **All models**: < 5 seconds

## Model Retraining

Models are automatically retrained when:
1. They're older than `MODEL_RETRAIN_INTERVAL_DAYS` (default: 7 days)
2. You explicitly request retraining via API
3. Significant data changes are detected

Configure in `.env`:
```env
MODEL_RETRAIN_INTERVAL_DAYS=7
```

## Storage Requirements

Per product:
- Prophet: ~1 MB
- LSTM: ~8 MB
- Random Forest: ~15 MB
- **Total per product**: ~25 MB

For 100 products: ~2.5 GB
For 1000 products: ~25 GB

## Cleanup

To remove old models:

```bash
# Remove all models (they'll be retrained on next forecast)
rm -rf ml_models/forecasting/*

# Remove models for specific product
rm -rf ml_models/forecasting/1/
```

## Verify Models

Check what models exist:

```bash
# List all trained models
find ml_models/forecasting -name "*.pkl" -o -name "*.h5"

# Check model info for product 1
cat ml_models/forecasting/1/model_info.json
```

## Configuration

Model settings in `config/settings.py`:

```python
FORECASTING_MODELS_PATH = "./ml_models/forecasting"
MODEL_RETRAIN_INTERVAL_DAYS = 7
MIN_TRAINING_SAMPLES = 100
```

## Best Practices

1. **First Run**: Let models train overnight for all products
2. **Disk Space**: Monitor disk usage as you add more products
3. **Backups**: Include `ml_models/` in your backup strategy
4. **Version Control**: Add `ml_models/` to `.gitignore` (models are large)

## Troubleshooting

### Models Not Saving
- Check disk space: `df -h`
- Verify permissions: `ls -la ml_models/`
- Check logs: `docker-compose logs api`

### Training Failures
- Verify sufficient historical data (100+ records)
- Check memory availability
- Review error logs

### Slow Predictions
- Ensure models are saved (not retraining each time)
- Check model file sizes are reasonable
- Consider using only Prophet for faster predictions

## Notes

- Models are product-specific (one set per product)
- Models persist across API restarts
- Docker volumes preserve models between container restarts
- Models are **not included in git** (too large)
- You can copy models between environments

## Quick Start

Don't worry about this folder being empty initially. Just run your first forecast and watch it populate automatically!

```bash
# This will create models for product 1
curl -X POST "http://localhost:8000/api/v1/forecast" \
  -H "Content-Type: application/json" \
  -d '{"product_id": 1, "forecast_horizon": 90}'
```

Then check:
```bash
ls -lh ml_models/forecasting/1/
```

You should see your trained models!
