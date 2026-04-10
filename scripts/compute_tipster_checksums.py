#!/usr/bin/env python3
"""Compute GlobChecker checksums for all TIPSTER collections.

Usage:
    python scripts/compute_tipster_checksums.py /path/to/TIPSTER
"""

import hashlib
import sys
from pathlib import Path


def compute_glob_checksum(folder: Path, glob: str) -> tuple[str, int]:
    """Compute the combined MD5 matching GlobChecker.compute() logic."""
    files = sorted(folder.glob(glob))
    files = [f for f in files if f.is_file()]
    combined = hashlib.md5()
    for f in files:
        combined.update(hashlib.md5(f.read_bytes()).hexdigest().encode())
    return combined.hexdigest(), len(files)


# (dataset_class, disk_path, glob_pattern)
COLLECTIONS = [
    ("Ap88", "Disk2/AP", "AP*"),
    ("Ap89", "Disk1/AP", "AP*"),
    ("Ap90", "Disk3/AP", "AP*"),
    ("Doe1", "Disk1/DOE", "DOE*"),
    ("Wsj87", "Disk1/WSJ/1987", "WSJ*"),
    ("Wsj88", "Disk1/WSJ/1988", "WSJ*"),
    ("Wsj89", "Disk1/WSJ/1989", "WSJ*"),
    ("Wsj90", "Disk2/WSJ/1990", "WSJ*"),
    ("Wsj91", "Disk2/WSJ/1991", "WSJ*"),
    ("Wsj92", "Disk2/WSJ/1992", "WSJ*"),
    ("Fr88", "Disk2/FR", "FR*"),
    ("Fr89", "Disk1/FR", "FR*"),
    ("Fr94", "Disk4/FR94", "**/*"),
    ("Ziff1", "Disk1/ZIFF", "ZF*"),
    ("Ziff2", "Disk2/ZIFF", "ZF*"),
    ("Ziff3", "Disk3/ZIFF", "ZF*"),
    ("Sjm1", "Disk3/SJM", "SJM*"),
    ("Cr1", "Disk4/CR", "**/*"),
    ("Ft1", "Disk4/FT", "**/*"),
    ("Fbis1", "Disk5/FBIS", "FB*"),
    ("La8990", "Disk5/LATIMES", "LA*"),
]


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <TIPSTER_DIR>", file=sys.stderr)
        sys.exit(1)

    tipster = Path(sys.argv[1])
    if not tipster.is_dir():
        print(f"Error: {tipster} is not a directory", file=sys.stderr)
        sys.exit(1)

    for name, disk_path, glob in COLLECTIONS:
        folder = tipster / disk_path
        if not folder.is_dir():
            print(f"WARNING: {folder} not found, skipping {name}", file=sys.stderr)
            continue
        md5, count = compute_glob_checksum(folder, glob)
        print(f'{name:10s}  glob={glob!r:10s}  files={count:4d}  md5="{md5}"  ({disk_path})')


if __name__ == "__main__":
    main()
