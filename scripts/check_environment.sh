#!/usr/bin/env bash
# ==============================================================================
# LUMI Vision - Environment Diagnostic & Verification Script
# Target: Ubuntu 24.04 LTS / Raspberry Pi 5 / Linux
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

ERRORS=0
WARNINGS=0

pass() { echo -e "  ${GREEN}[PASS]${NC} $1"; }
warn() { echo -e "  ${YELLOW}[WARN]${NC} $1"; ((WARNINGS++)); }
fail() { echo -e "  ${RED}[FAIL]${NC} $1"; ((ERRORS++)); }
info() { echo -e "  ${CYAN}[INFO]${NC} $1"; }

echo -e "\n${BOLD}====================================================${NC}"
echo -e "${BOLD}       LUMI Vision - Environment Diagnostic        ${NC}"
echo -e "${BOLD}====================================================${NC}\n"

# 1. OS & Kernel Check
echo -e "${BOLD}1. System & Architecture:${NC}"
OS_NAME=$(uname -s)
ARCH=$(uname -m)
info "Kernel: $OS_NAME $(uname -r) ($ARCH)"

if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" && "${VERSION_ID:-}" == "24.04" ]]; then
        pass "Operating System: $PRETTY_NAME"
    elif [[ "$ID" == "ubuntu" ]]; then
        warn "Operating System: $PRETTY_NAME (Ubuntu 24.04 LTS recommended)"
    else
        warn "Operating System: $PRETTY_NAME (Target is Ubuntu 24.04 LTS)"
    fi
else
    warn "Cannot identify Linux distribution (non-standard /etc/os-release)"
fi

# Check Architecture
if [[ "$ARCH" == "x86_64" || "$ARCH" == "aarch64" ]]; then
    pass "CPU Architecture: $ARCH (Supported)"
else
    warn "CPU Architecture: $ARCH (Untested)"
fi

# 2. Python Check
echo -e "\n${BOLD}2. Python Runtime:${NC}"
PY_CMD=""
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    info "Active Virtualenv: $VIRTUAL_ENV"
    PY_CMD="python"
elif [[ -f "$REPO_ROOT/.venv/bin/python" ]]; then
    info "Found local .venv environment"
    PY_CMD="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY_CMD="python3"
fi

if [[ -n "$PY_CMD" ]]; then
    PY_VERSION=$("$PY_CMD" -c "import sys; print('.'.join(map(str, sys.version_info[:3])))")
    PY_MAJOR=$("$PY_CMD" -c "import sys; print(sys.version_info.major)")
    PY_MINOR=$("$PY_CMD" -c "import sys; print(sys.version_info.minor)")
    if [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -eq 12 ]]; then
        pass "Python version: $PY_VERSION (Matches Ubuntu 24.04 default Python 3.12)"
    elif [[ "$PY_MAJOR" -eq 3 && "$PY_MINOR" -ge 10 ]]; then
        pass "Python version: $PY_VERSION (Compatible Python 3.$PY_MINOR)"
    else
        fail "Python version: $PY_VERSION (Requires Python >= 3.10)"
    fi
else
    fail "No Python executable located!"
fi

# 3. Python Dependencies
echo -e "\n${BOLD}3. Python Packages:${NC}"
if [[ -n "$PY_CMD" ]]; then
    check_package() {
        local pkg="$1"
        local import_name="$2"
        if "$PY_CMD" -c "import $import_name" >/dev/null 2>&1; then
            local ver
            ver=$("$PY_CMD" -c "import $import_name; print(getattr($import_name, '__version__', 'available'))")
            pass "$pkg (v$ver)"
        else
            fail "Package '$pkg' is NOT installed"
        fi
    }

    check_package "OpenCV" "cv2"
    check_package "NumPy" "numpy"
    check_package "PyYAML" "yaml"
    check_package "pytest" "pytest"
else
    fail "Cannot check Python packages because Python executable is missing"
fi

# 4. Vision Models
echo -e "\n${BOLD}4. Pretrained Vision Models:${NC}"
YUNET_PATH="$REPO_ROOT/models/face_detection_yunet_2023mar.onnx"
SFACE_PATH="$REPO_ROOT/models/face_recognition_sface_2021dec.onnx"

YUNET_HASH="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
SFACE_HASH="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"

compute_hash() {
    local f="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$f" | awk '{print $1}'
    elif [[ -n "$PY_CMD" ]]; then
        "$PY_CMD" -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" "$f"
    else
        echo "unknown"
    fi
}

if [[ -f "$YUNET_PATH" ]]; then
    H=$(compute_hash "$YUNET_PATH")
    if [[ "$H" == "$YUNET_HASH" ]]; then
        pass "YuNet Face Detector model verified ($YUNET_PATH)"
    else
        fail "YuNet model corrupt or hash mismatch: $H"
    fi
else
    fail "YuNet model missing at $YUNET_PATH (Run bash scripts/download_models.sh)"
fi

if [[ -f "$SFACE_PATH" ]]; then
    H=$(compute_hash "$SFACE_PATH")
    if [[ "$H" == "$SFACE_HASH" ]]; then
        pass "SFace Face Recognizer model verified ($SFACE_PATH)"
    else
        fail "SFace model corrupt or hash mismatch: $H"
    fi
else
    fail "SFace model missing at $SFACE_PATH (Run bash scripts/download_models.sh)"
fi

# 5. Camera Device Checks
echo -e "\n${BOLD}5. Camera Hardware Accessibility:${NC}"
VIDEO_DEVICES=(/dev/video*)
if [[ -e "${VIDEO_DEVICES[0]}" ]]; then
    pass "Video devices detected in /dev: ${VIDEO_DEVICES[*]}"
    for dev in "${VIDEO_DEVICES[@]}"; do
        if [[ -r "$dev" && -w "$dev" ]]; then
            pass "Device permissions OK for $dev"
        else
            warn "Permission restricted for $dev. User may need to be added to 'video' group (sudo usermod -a -G video \$USER)"
        fi
    done
else
    warn "No /dev/video* devices found. Webcam may be disconnected or running inside container/VM/WSL without USB passthrough."
fi

# 6. ROS 2 Environment Checks (Optional for Standalone, Required for Robot)
echo -e "\n${BOLD}6. ROS 2 Environment (Target: ROS 2 Jazzy):${NC}"
if [[ -n "${ROS_DISTRO:-}" ]]; then
    if [[ "$ROS_DISTRO" == "jazzy" ]]; then
        pass "ROS 2 Jazzy is active (ROS_DISTRO=jazzy)"
    else
        info "ROS 2 is active with distribution: $ROS_DISTRO (Target is Jazzy)"
    fi
elif [[ -d "/opt/ros/jazzy" ]]; then
    info "ROS 2 Jazzy is installed at /opt/ros/jazzy (not yet sourced in current shell)"
    info "To activate ROS 2: source /opt/ros/jazzy/setup.bash"
else
    info "ROS 2 Jazzy not detected. Note: Standalone vision module operates independently without ROS 2."
fi

# Summary
echo -e "\n${BOLD}====================================================${NC}"
if [[ $ERRORS -eq 0 && $WARNINGS -eq 0 ]]; then
    echo -e "${BOLD}${GREEN}  DIAGNOSTIC STATUS: READY (0 Errors, 0 Warnings)  ${NC}"
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${BOLD}${YELLOW}  DIAGNOSTIC STATUS: PASS (0 Errors, $WARNINGS Warnings) ${NC}"
else
    echo -e "${BOLD}${RED}  DIAGNOSTIC STATUS: ACTION REQUIRED ($ERRORS Errors, $WARNINGS Warnings) ${NC}"
fi
echo -e "${BOLD}====================================================${NC}\n"

exit $ERRORS
