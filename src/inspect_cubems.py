"""
inspect_cubems.py
=================
Inspect CUBEMS 2019 Floor 2 và Floor 5.

Mục đích:
  1. Xác nhận số cột, tên cột, kiểu dữ liệu
  2. Thống kê missing values theo cột
  3. Xem phân phối power của từng zone
  4. Xác định cấu trúc để lên kế hoạch resampling (1min -> hourly)
"""
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/raw")

FILES = {
    "Floor2": "2019Floor2.csv",
    "Floor5": "2019Floor5.csv",
}


def inspect_file(name: str, filename: str):
    path = DATA_DIR / filename
    print(f"\n{'='*72}")
    print(f"FILE: {filename}  ({name})")
    print(f"{'='*72}")

    # Đọc với parse_dates
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"Shape: {df.shape}")
    print(f"Time range: {df['Date'].min()} -> {df['Date'].max()}")
    print(f"Số cột: {len(df.columns)}")

    # Nhóm cột theo zone
    print(f"\n--- Cột theo Zone ---")
    zones = {}
    for col in df.columns:
        if col == "Date":
            continue
        # Pattern: z1_AC1(kW), z2_Light(kW), etc.
        zone_prefix = col.split("_")[0]  # "z1", "z2", ...
        zones.setdefault(zone_prefix, []).append(col)

    for zone, cols in sorted(zones.items()):
        print(f"\n  {zone}: {len(cols)} cột")
        for c in cols:
            non_null = df[c].notna().sum()
            pct_missing = 100 * (1 - non_null / len(df))
            print(f"    - {c:25s}  missing: {pct_missing:5.2f}%")

    # Thống kê power của từng zone (sum AC + Light + Plug)
    print(f"\n--- Zone-level power statistics (kW) ---")
    for zone in sorted(zones.keys()):
        # Tìm các cột power
        power_cols = [c for c in zones[zone] if "(kW)" in c]
        if not power_cols:
            continue
        zone_power = df[power_cols].sum(axis=1, skipna=True)
        print(f"\n  {zone} total power ({len(power_cols)} cols):")
        print(f"    min:    {zone_power.min():.3f}")
        print(f"    max:    {zone_power.max():.3f}")
        print(f"    mean:   {zone_power.mean():.3f}")
        print(f"    median: {zone_power.median():.3f}")
        print(f"    std:    {zone_power.std():.3f}")

    return df


if __name__ == "__main__":
    for name, filename in FILES.items():
        inspect_file(name, filename)