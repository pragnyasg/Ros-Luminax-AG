#!/usr/bin/env bash
# ==============================================================================
# LUMI Vision - Setup Script for Ubuntu 24.04 LTS (x86_64 and aarch64 / Pi 5)
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BOLD}${BLUE}====================================================================${NC}"
echo -e "${BOLD}${BLUE}         LUMI Vision - Ubuntu 24.04 LTS Automated Setup             ${NC}"
echo -e "${BOLD}${BLUE}====================================================================${NC}"

# 1. System & Architecture Inspection
echo -e "\n${BOLD}[1/6] Inspecting System Environment...${NC}"
ARCH=$(uname -m)
echo "Operating System : $(uname -s)"
echo "Kernel Release   : $(uname -r)"
echo "Architecture     : $ARCH"

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    echo "Linux Distro     : $PRETTY_NAME"
    if [[ "$ID" == "ubuntu" && "${VERSION_ID:-}" != "24.04" ]]; then
        echo -e "${YELLOW}[NOTE] Designed for Ubuntu 24.04 LTS, detected $VERSION_ID. Proceeding...${NC}"
    fi
fi

# 2. Check Python 3.12+
echo -e "\n${BOLD}[2/6] Checking Python Version...${NC}"
PYTHON_BIN=""
for candidate in python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        PY_VER=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        PY_MAJOR=$("$candidate" -c "import sys; print(sys.version_info.major)")
        PY_MINOR=$("$candidate" -c "import sys; print(sys.version_info.minor)")
        if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 10 ]]; then
            PYTHON_BIN="$candidate"
            echo -e "${GREEN}[OK] Found compatible Python: $PYTHON_BIN (v$PY_VER)${NC}"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    echo -e "${RED}[ERROR] Python 3.10+ (preferably 3.12) is required but was not found.${NC}"
    echo "Please install Python 3.12 using: sudo apt update && sudo apt install -y python3.12 python3.12-venv python3-pip"
    exit 1
fi

# 3. Check for virtual environment tool
echo -e "\n${BOLD}[3/6] Setting Up Isolated Virtual Environment...${NC}"
VENV_DIR="$REPO_ROOT/.venv"

if ! "$PYTHON_BIN" -m venv --help >/dev/null 2>&1; then
    echo -e "${RED}[ERROR] python3-venv is missing on this system.${NC}"
    echo "Ubuntu 24.04 requires the python3-venv package to create isolated environments without modifying system Python."
    echo "Run: sudo apt update && sudo apt install -y python3-venv python3.12-venv"
    exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment at $VENV_DIR..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo -e "${GREEN}[OK] Virtual environment created.${NC}"
else
    echo -e "${GREEN}[OK] Virtual environment already exists at $VENV_DIR.${NC}"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"

# Ensure pip inside venv is functional
"$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
"$VENV_PIP" install --upgrade pip setuptools wheel >/dev/null

# 4. Install Dependencies
echo -e "\n${BOLD}[4/6] Installing Python Dependencies in .venv...${NC}"
echo "Installing from: $REPO_ROOT/requirements.txt"
"$VENV_PIP" install -r "$REPO_ROOT/requirements.txt"
"$VENV_PIP" install -e "$REPO_ROOT"
echo -e "${GREEN}[OK] Dependencies and luminax_vision package installed successfully.${NC}"

# 5. Configuration Setup
echo -e "\n${BOLD}[5/6] Verifying Configuration...${NC}"
EXAMPLE_CONFIG="$REPO_ROOT/config/config.example.yaml"
ACTIVE_CONFIG="$REPO_ROOT/config/config.yaml"

if [[ ! -f "$ACTIVE_CONFIG" ]]; then
    if [[ -f "$EXAMPLE_CONFIG" ]]; then
        echo "Creating initial $ACTIVE_CONFIG from template..."
        cp "$EXAMPLE_CONFIG" "$ACTIVE_CONFIG"
        echo -e "${GREEN}[OK] Created $ACTIVE_CONFIG.${NC}"
    else
        echo -e "${RED}[ERROR] Template $EXAMPLE_CONFIG not found!${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}[OK] Configuration $ACTIVE_CONFIG already exists. Preserving existing settings.${NC}"
fi

# Ensure data directory exists
mkdir -p "$REPO_ROOT/data"

# 6. Download Pretrained Models
echo -e "\n${BOLD}[6/6] Verifying and Downloading ONNX Models...${NC}"
bash "$REPO_ROOT/scripts/download_models.sh"

echo -e "\n${BOLD}${GREEN}====================================================================${NC}"
echo -e "${BOLD}${GREEN}        LUMI Vision Setup Completed Successfully!                   ${NC}"
echo -e "${BOLD}${GREEN}====================================================================${NC}"
echo -e "To activate the environment and launch the standalone module, run:"
echo -e "\n  ${BOLD}source .venv/bin/activate${NC}"
echo -e "  ${BOLD}python -m luminax_vision${NC}\n"
