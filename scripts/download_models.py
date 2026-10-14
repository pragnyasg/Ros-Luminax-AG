"""
Cross-platform model downloader with SHA-256 verification for LUMI Vision.
Works on Ubuntu Linux, macOS, and Windows.
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

MODELS = [
    {
        "name": "YuNet Face Detector",
        "filename": "face_detection_yunet_2023mar.onnx",
        "sha256": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        "urls": [
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    },
    {
        "name": "SFace Face Recognizer",
        "filename": "face_recognition_sface_2021dec.onnx",
        "sha256": "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        "urls": [
            "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
    },
]


def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_with_progress(url: str, dest_path: Path) -> None:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "LUMI-Vision-ModelDownloader/1.0"},
    )
    with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 65536

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            downloaded += len(chunk)
            out_file.write(chunk)
            if total_size > 0:
                percent = downloaded / total_size * 100
                mb_down = downloaded / (1024 * 1024)
                mb_tot = total_size / (1024 * 1024)
                sys.stdout.write(f"\r  Downloading: {mb_down:.1f}/{mb_tot:.1f} MB ({percent:.1f}%)")
            else:
                sys.stdout.write(f"\r  Downloaded: {downloaded / (1024 * 1024):.1f} MB")
            sys.stdout.flush()
        print()


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    models_dir = repo_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("====================================================")
    print("      LUMI Vision - Model Download & Verification   ")
    print("====================================================")

    for item in MODELS:
        dest = models_dir / item["filename"]
        expected_hash = item["sha256"]
        print(f"\nChecking {item['name']} ({item['filename']})...")

        if dest.is_file():
            current_hash = compute_sha256(dest)
            if current_hash == expected_hash:
                print(f"  [OK] Already present and verified: {dest.name}")
                continue
            else:
                print("  [WARN] Checksum mismatch on existing file. Re-downloading...")
                dest.unlink()

        success = False
        temp_dest = dest.with_suffix(".tmp")
        for url in item["urls"]:
            print(f"  Connecting to: {url}")
            try:
                download_with_progress(url, temp_dest)
                print("  Verifying SHA-256 checksum...")
                downloaded_hash = compute_sha256(temp_dest)
                if downloaded_hash == expected_hash:
                    temp_dest.replace(dest)
                    print(f"  [SUCCESS] Verified and saved to: {dest.name}")
                    success = True
                    break
                else:
                    print(f"  [ERROR] Checksum failed for source: {url}")
                    print(f"    Expected: {expected_hash}")
                    print(f"    Got:      {downloaded_hash}")
                    if temp_dest.exists():
                        temp_dest.unlink()
            except Exception as err:
                print(f"  [WARN] Download failed from {url}: {err}")
                if temp_dest.exists():
                    temp_dest.unlink()

        if not success:
            print(f"  [FATAL] Failed to download {item['filename']} from all available sources.")
            sys.exit(1)

    print("\nAll required models downloaded and verified successfully!\n")


if __name__ == "__main__":
    main()
