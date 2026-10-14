#!/usr/bin/env bash
# ==============================================================================
# LUMI Vision - Automated Model Downloader & Checksum Verifier
# Target: Ubuntu 24.04 LTS / Linux / macOS
# ==============================================================================

set -euo pipefail

# Determine script and repository root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODELS_DIR="$REPO_ROOT/models"

mkdir -p "$MODELS_DIR"

# Text formatting
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BOLD}====================================================${NC}"
echo -e "${BOLD}       LUMI Vision - Model Download & Verification  ${NC}"
echo -e "${BOLD}====================================================${NC}"

# Model 1: YuNet Face Detection
YUNET_FILE="face_detection_yunet_2023mar.onnx"
YUNET_SHA256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_URL_PRIMARY="https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx"
YUNET_URL_FALLBACK="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# Model 2: SFace Face Recognition
SFACE_FILE="face_recognition_sface_2021dec.onnx"
SFACE_SHA256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SFACE_URL_PRIMARY="https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx"
SFACE_URL_FALLBACK="https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Helper: Compute SHA-256
compute_sha256() {
    local target_file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$target_file" | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$target_file" | awk '{print $1}'
    elif command -v python3 >/dev/null 2>&1; then
        python3 -c "import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], 'rb').read()).hexdigest())" "$target_file"
    else
        echo "unknown"
    fi
}

# Helper: Download file using curl or wget
download_file() {
    local url="$1"
    local output="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -f -L --retry 3 --retry-delay 2 -o "$output" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --waitretry=2 -O "$output" "$url"
    else
        echo -e "${RED}Error: Neither curl nor wget is installed.${NC}" >&2
        return 1
    fi
}

download_and_verify() {
    local name="$1"
    local filename="$2"
    local expected_hash="$3"
    local primary_url="$4"
    local fallback_url="$5"
    local dest_path="$MODELS_DIR/$filename"

    echo -e "\n--> Checking ${BOLD}$name${NC} ($filename)..."

    if [[ -f "$dest_path" ]]; then
        local current_hash
        current_hash=$(compute_sha256 "$dest_path")
        if [[ "$current_hash" == "$expected_hash" ]]; then
            echo -e "${GREEN}[OK] Already present and verified:${NC} $filename"
            return 0
        else
            echo -e "${YELLOW}[WARN] Existing file failed checksum verification. Re-downloading...${NC}"
            rm -f "$dest_path"
        fi
    fi

    echo "Downloading from primary source: $primary_url"
    local temp_dest="${dest_path}.tmp"
    if ! download_file "$primary_url" "$temp_dest"; then
        echo -e "${YELLOW}[WARN] Primary source failed. Trying fallback source...${NC}"
        download_file "$fallback_url" "$temp_dest" || {
            echo -e "${RED}[ERROR] Failed to download $filename from all sources.${NC}"
            rm -f "$temp_dest"
            return 1
        }
    fi

    echo "Verifying SHA-256 checksum..."
    local downloaded_hash
    downloaded_hash=$(compute_sha256 "$temp_dest")

    if [[ "$downloaded_hash" != "$expected_hash" ]]; then
        echo -e "${RED}[ERROR] Checksum mismatch for $filename!${NC}"
        echo "Expected: $expected_hash"
        echo "Got:      $downloaded_hash"
        rm -f "$temp_dest"
        return 1
    fi

    mv "$temp_dest" "$dest_path"
    echo -e "${GREEN}[SUCCESS] Downloaded and verified:${NC} $dest_path"
    return 0
}

# Download YuNet and SFace
download_and_verify "YuNet Face Detector" "$YUNET_FILE" "$YUNET_SHA256" "$YUNET_URL_PRIMARY" "$YUNET_URL_FALLBACK"
download_and_verify "SFace Face Recognizer" "$SFACE_FILE" "$SFACE_SHA256" "$SFACE_URL_PRIMARY" "$SFACE_URL_FALLBACK"

echo -e "\n${BOLD}${GREEN}All required vision models are downloaded and verified successfully!${NC}\n"
