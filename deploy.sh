#!/bin/bash
# ════════════════════════════════════════════════════════════
# NSE ALGO TRADER — Production Deployment Script
# ════════════════════════════════════════════════════════════
#
# This script handles:
# 1. Environment setup
# 2. Dependency installation
# 3. Configuration validation
# 4. Database initialization
# 5. Service installation (systemd)
# 6. Dashboard launch
#
# Usage:
#   ./deploy.sh [dev|prod]
#
# ════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"
CONFIG_DIR="$SCRIPT_DIR/config"
LOG_DIR="$SCRIPT_DIR/logs"
DATA_DIR="$SCRIPT_DIR/data"
ENV="${1:-dev}"

echo -e "${BLUE}"
echo "════════════════════════════════════════════════════════════"
echo "  NSE ALGO TRADER — Deployment Script"
echo "  Environment: $ENV"
echo "════════════════════════════════════════════════════════════"
echo -e "${NC}"

# ────────────────────────────────────────────────────────────
# Step 1: Check prerequisites
# ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/7] Checking prerequisites...${NC}"

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  Python version: $PYTHON_VERSION"

if [[ $(echo "$PYTHON_VERSION < 3.9" | bc -l) -eq 1 ]]; then
    echo -e "${RED}Error: Python 3.9+ required${NC}"
    exit 1
fi

# Check pip
if ! command -v pip3 &> /dev/null; then
    echo -e "${RED}Error: pip3 is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}  ✓ Prerequisites OK${NC}"

# ────────────────────────────────────────────────────────────
# Step 2: Create directories
# ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/7] Creating directories...${NC}"

mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$DATA_DIR/cache"
mkdir -p "$DATA_DIR/backups"
mkdir -p "$SCRIPT_DIR/results"
mkdir -p "$SCRIPT_DIR/models"

echo -e "${GREEN}  ✓ Directories created${NC}"

# ────────────────────────────────────────────────────────────
# Step 3: Setup virtual environment
# ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/7] Setting up virtual environment...${NC}"

if [ ! -d "$VENV_DIR" ]; then
    echo "  Creating new virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "  Upgrading pip..."
pip install --upgrade pip -q

echo -e "${GREEN}  ✓ Virtual environment ready${NC}"

# ────────────────────────────────────────────────────────────
# Step 4: Install dependencies
# ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/7] Installing dependencies...${NC}"

pip install -r "$SCRIPT_DIR/requirements.txt" -q

# Additional production dependencies
pip install psutil gunicorn -q

echo -e "${GREEN}  ✓ Dependencies installed${NC}"

# ────────────────────────────────────────────────────────────
# Step 5: Validate configuration
# ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[5/7] Validating configuration...${NC}"

if [ "$ENV" == "prod" ]; then
    CONFIG_FILE="$CONFIG_DIR/config_prod.yaml"
else
    CONFIG_FILE="$CONFIG_DIR/config_test.yaml"
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG_FILE${NC}"
    echo "  Please copy config_example.yaml and configure:"
    echo "    cp $CONFIG_DIR/config_example.yaml $CONFIG_FILE"
    exit 1
fi

# Validate required keys
python3 << EOF
import yaml
import sys

with open("$CONFIG_FILE") as f:
    config = yaml.safe_load(f)

required = ["broker", "capital", "telegram"]
missing = [k for k in required if k not in config]

if missing:
    print(f"Missing required config keys: {missing}")
    sys.exit(1)

# Check broker config
broker = config.get("broker", {}).get("angel_one", {})
if not broker.get("api_key"):
    print("Warning: Angel One API key not configured")

# Check telegram
tg = config.get("telegram", {})
if not tg.get("bot_token") or not tg.get("chat_ids"):
    print("Warning: Telegram not fully configured")

print("  Configuration valid")
EOF

echo -e "${GREEN}  ✓ Configuration validated${NC}"

# ────────────────────────────────────────────────────────────
# Step 6: Initialize database
# ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[6/7] Initializing database...${NC}"

python3 -c "
from core.trade_database import TradeDB
db = TradeDB()
print('  Database initialized')
"

echo -e "${GREEN}  ✓ Database ready${NC}"

# ────────────────────────────────────────────────────────────
# Step 7: Test imports
# ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[7/7] Testing imports...${NC}"

python3 << EOF
import sys
try:
    from live_paper_v3 import run
    from strategies.claude_brain_v2 import ClaudeBrainV2
    from strategies.pro_strategy_v2 import ProStrategyV2
    from data.angel_broker import AngelBroker
    from core.trade_database import TradeDB
    from core.system_monitor import SystemMonitor, KillSwitch
    print("  All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
EOF

echo -e "${GREEN}  ✓ All imports OK${NC}"

# ────────────────────────────────────────────────────────────
# Deployment complete
# ────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════"
echo "  DEPLOYMENT SUCCESSFUL!"
echo "════════════════════════════════════════════════════════════${NC}"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Configure API keys in $CONFIG_FILE"
echo ""
echo "  2. Test the system:"
echo "     source .venv/bin/activate"
echo "     python test_apis.py"
echo ""
echo "  3. Run paper trading:"
echo "     python live_paper_v3.py"
echo ""
echo "  4. Start dashboard:"
echo "     streamlit run dashboard/production_dashboard.py"
echo ""
echo "  5. (Optional) Install as service:"
echo "     sudo cp nse_algo_trader.service /etc/systemd/system/"
echo "     sudo systemctl enable nse_algo_trader"
echo "     sudo systemctl start nse_algo_trader"
echo ""
echo -e "${BLUE}Happy Trading! 📈${NC}"
