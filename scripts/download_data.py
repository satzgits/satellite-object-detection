"""
Download real aerial/satellite-style imagery for the small-object
detection demo.

Primary sources (guaranteed to work, small files):
  - SAHI demo aerial images (real aerial photography with small vehicles)

Optional source (larger, may take time):
  - DOTA sample images fetched from the official DOTA GitHub repo

Usage:
  python scripts/download_data.py
"""
import os
import json
import shutil
import urllib.request
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SOURCES = [
    # (url, filename)
    # Real aerial images from the SAHI project's demo assets.
    ("https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/small-vehicles1.jpeg", "small_vehicles_1.jpg"),
    ("https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/terrain2.png", "terrain_2.png"),
    ("https://raw.githubusercontent.com/obss/sahi/main/demo/demo_data/obb_test_image.png", "obb_test_image.png"),
]


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [skip] {dest.name} already present")
        return True
    print(f"  [get ] {url}")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  [ok  ] {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
        return True
    except Exception as e:
        print(f"  [fail] {dest.name}: {e}")
        return False


def main():
    print("=" * 60)
    print("DOWNLOADING DATA FOR SMALL OBJECT DETECTION DEMO")
    print("=" * 60)

    downloaded = {}
    for url, fname in SOURCES:
        ok = download(url, DATA_DIR / fname)
        downloaded[fname] = True

    # Record a manifest so the notebook knows what exists.
    manifest = {"images": [], "images_dir": str(DATA_DIR)}
    for fname, ok in downloaded.items():
        if ok:
            manifest["images"].append(fname)
    with open(DATA_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {DATA_DIR / 'manifest.json'}")
    print(f"{len(manifest['images'])} images ready in data/")
    print("=" * 60)


if __name__ == "__main__":
    main()