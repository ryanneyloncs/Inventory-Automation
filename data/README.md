# Data Directories

## Structure

```
data/
├── raw/        - Raw data files (CSV, Excel, JSON)
└── processed/  - Processed/cleaned data ready for use
```

## Raw Data

Place your source data files here:
- Sales records (CSV, Excel)
- Product catalogs
- Supplier information
- Historical inventory data

**Supported formats:**
- CSV (`.csv`)
- Excel (`.xlsx`, `.xls`)
- JSON (`.json`)
- TSV (`.tsv`)

## Processed Data

Cleaned and transformed data stored here:
- Normalized sales data
- Feature-engineered datasets
- Training/validation splits
- Aggregated statistics

## Usage

### Import Sales Data

```python
import pandas as pd
from scripts.data_import import import_sales_data

# Load CSV
df = pd.read_csv('data/raw/sales_2024.csv')

# Import to database
await import_sales_data(df)
```

### Export Data

```python
# Export current inventory
from scripts.data_export import export_inventory

await export_inventory('data/processed/inventory_snapshot.csv')
```

## Data Schema

### Sales Data Format
```csv
product_id,sale_date,quantity,revenue,channel
1,2024-01-01,10,299.90,online
2,2024-01-01,5,149.95,retail
```

### Product Data Format
```csv
sku,name,category,unit_cost,selling_price,supplier_id
WDG-001,Premium Widget,Widgets,15.50,29.99,1
GAD-001,Ultra Gadget,Gadgets,45.00,89.99,2
```

## Security

- Add `data/` to `.gitignore` (already done)
- Don't commit customer data
- Use data encryption for sensitive files
- Regular backups recommended

## Cleanup

```bash
# Remove old raw files
rm data/raw/*.csv

# Clear processed cache
rm data/processed/*
```

## Data Pipeline

1. **Extract**: Place raw files in `data/raw/`
2. **Transform**: Process with data cleaning scripts
3. **Load**: Import to database via API or scripts
4. **Archive**: Move processed files to `data/processed/`

## Notes

- These directories are empty by design
- Data is generated/imported at runtime
- Use for batch imports and exports
- Not for application runtime data (use database)
