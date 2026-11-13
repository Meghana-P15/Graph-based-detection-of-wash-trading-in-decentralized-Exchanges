#!/usr/bin/env python3
"""Download official Zenodo raw files (no fabricated trading data)."""
import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://zenodo.org/records/4540223/files/"
FILES = {
    "IDEXTrades.csv": "497b9c3622c47d51d7db1119fbd49b78",
    "EtherDeltaTrades.csv": "0c5c6c9a05c2833b82095503254fccb5",
    "token_decimals.json": "9edd0f4213fc9feaaf1a5303d004adce",
    "EtherDollarPrice.csv": "b47e050a7e3a7a828af19acd717c70eb",
}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(filename, expected_md5, directory):
    dest = directory / filename
    tmp = directory / (filename + ".part")
    if dest.is_file() and md5(dest) == expected_md5:
        print(f"[OK] {filename} already exists and checksum matches")
        return True
    if dest.exists():
        print(f"[WARN] {filename} exists but MD5 differs; leave file for inspection", file=sys.stderr)
        return False

    for attempt in range(1, 4):
        offset = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(BASE + filename + "?download=1", headers={
            "User-Agent": "AstraQuant-Research-Dataset-Fetch/1.0",
            **({"Range": f"bytes={offset}-"} if offset else {}),
        })
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                resumed = offset > 0 and getattr(resp, "status", 200) == 206
                mode = "ab" if resumed else "wb"
                if offset and not resumed:
                    print(f"[INFO] Server did not resume {filename}; restarting partial file")
                with open(tmp, mode) as out:
                    count = offset if resumed else 0
                    last = count
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        out.write(chunk)
                        count += len(chunk)
                        if count - last >= 64 * 1024 * 1024:
                            print(f"[DOWNLOADING] {filename}: {count/1e9:.2f} GB", flush=True)
                            last = count
            print(f"[CHECKING] {filename} ({tmp.stat().st_size/1e9:.3f} GB)")
            if md5(tmp) != expected_md5:
                print(f"[ERROR] MD5 failed for {filename}; leaving .part for inspection", file=sys.stderr)
                return False
            tmp.rename(dest)
            print(f"[DONE] {filename}")
            return True
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"[RETRY {attempt}/3] {filename}: {e}", file=sys.stderr)
            time.sleep(2 * attempt)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", default="data/raw/historical", type=Path)
    ap.add_argument("--files", nargs="+", choices=list(FILES), default=list(FILES))
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = []
    for filename in args.files:
        if not download(filename, FILES[filename], args.output):
            failures.append(filename)
    print(json.dumps({"verified": sorted(set(args.files) - set(failures)), "failed": failures}, indent=2))
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
