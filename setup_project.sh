#!/bin/bash

# Create full directory structure
mkdir -p src/{api,models,services,utils}
mkdir -p src/services/{forecasting,optimization,reorder,suppliers}
mkdir -p config
mkdir -p scripts
mkdir -p tests/{unit,integration}
mkdir -p data/{raw,processed}
mkdir -p logs
mkdir -p deployments/{docker,kubernetes}
mkdir -p ml_models

# Create __init__.py files
touch src/__init__.py
touch src/api/__init__.py
touch src/models/__init__.py
touch src/services/__init__.py
touch src/services/forecasting/__init__.py
touch src/services/optimization/__init__.py
touch src/services/reorder/__init__.py
touch src/services/suppliers/__init__.py
touch src/utils/__init__.py
touch tests/__init__.py
touch tests/unit/__init__.py
touch tests/integration/__init__.py

echo "Project structure created successfully!"
