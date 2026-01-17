# Installation Guide

## Quick Install

### 1. Install Core Dependencies

```bash
pip install pandas numpy pyyaml requests tabulate tqdm matplotlib
```

### 2. Install Data Source (Yahoo Finance)

For real market data, install yfinance:

```bash
# Try full install first
pip install yfinance

# If that fails due to multitasking/curl_cffi issues, try:
pip install --no-deps yfinance
pip install beautifulsoup4 html5lib peewee frozendict lxml appdirs platformdirs protobuf websockets
```

### 3. Verify Installation

```bash
python test_system.py
```

This will test the core components with synthetic data.

### 4. Test with Real Data (Optional)

If yfinance installed successfully:

```bash
python examples/quick_start.py
```

## Alternative Data Sources

If Yahoo Finance doesn't work, you can use:

### Polygon.io (Paid, Reliable)

```bash
export POLYGON_API_KEY="your_api_key"
```

Edit `config.yaml`:
```yaml
data_source:
  primary: "polygon"
```

### Alpha Vantage (Free Tier Available)

```bash
export ALPHAVANTAGE_API_KEY="your_api_key"
```

Edit `config.yaml`:
```yaml
data_source:
  primary: "alphavantage"
```

## Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# If yfinance fails, use the manual method above
```

## Docker (Alternative)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir pandas numpy pyyaml requests tabulate tqdm matplotlib

COPY . .

CMD ["python", "main.py", "--mode", "signals"]
```

Build and run:
```bash
docker build -t pytrade .
docker run -v $(pwd)/output:/app/output pytrade
```

## Troubleshooting

### Issue: yfinance multitasking error

**Solution**: Install without dependencies first, then install deps manually (see Quick Install step 2)

### Issue: pandas-ta not found

**Solution**: Not needed! We implemented indicators ourselves. Remove from requirements if present.

### Issue: Permission errors

**Solution**: Use virtual environment or add `--user` flag:
```bash
pip install --user -r requirements.txt
```

### Issue: SSL certificate errors

**Solution**:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

## Minimal Setup (No External Data)

For testing/development without live data:

```bash
pip install pandas numpy pyyaml tabulate
python test_system.py
```

This installs only core dependencies and uses synthetic data.

## System Requirements

- Python 3.10 or higher
- 500MB free disk space
- Internet connection for data fetching
- 2GB RAM minimum (4GB recommended for backtesting)

## Platform-Specific Notes

### Linux
```bash
# Debian/Ubuntu may need:
sudo apt-get install python3-dev build-essential

# For matplotlib:
sudo apt-get install python3-tk
```

### Mac
```bash
# Use Homebrew Python
brew install python@3.11
```

### Windows
```bash
# Use official Python installer from python.org
# Add Python to PATH during installation
```

## Verifying Installation

Test each component:

```python
# Test core imports
python -c "import pandas, numpy, yaml, tabulate; print('Core: OK')"

# Test our modules
python -c "import sys; sys.path.insert(0, 'src'); from indicators import TechnicalIndicators; print('Modules: OK')"

# Test data fetching (requires yfinance)
python -c "import yfinance; print('yfinance: OK')"

# Run full test
python test_system.py
```

All tests passing? You're ready to go!

```bash
python main.py --mode signals
```
