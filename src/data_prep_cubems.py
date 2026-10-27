"""
Preprocessing cho CUBEMS 2019 Floor 2 và Floor 5.

==============================================================================
MỤC ĐÍCH
==============================================================================

Chuyển 2 files CSV thô (37 cột × 525,600 dòng mỗi file, sampling 1 phút)
thành 6 files sạch (6 features × 8,760 dòng mỗi file, sampling 1 giờ).

Mỗi output file = 1 ZONE (thay vì 1 appliance như dataset cũ).

==============================================================================
CHIẾN LƯỢC
==============================================================================

1. Chọn 6 zones:
   - floor2_z1: 1 AC, power mean 11.79 kW
   - floor2_z2: 14 AC, power mean 12.26 kW (đặc biệt)
   - floor2_z4: 1 AC, power mean 2.63 kW
   - floor5_z1: 4 AC, power mean 7.03 kW
   - floor5_z4: 1 AC, power mean 5.71 kW
   - floor5_z5: 1 AC, power mean 2.84 kW

2. Gộp power: zone_power = Σ(all (kW) columns in zone)

3. Resample 1 phút → 1 giờ bằng mean aggregation.

4. Context features: hour (derived), temperature, humidity, lux.


==============================================================================
FEATURE VECTOR OUTPUT (giống hệt CLEF Sharjah)
==============================================================================

    [power, hour, day_of_week, temperature, humidity, lux]

Context vector:
    [hour, day_of_week, temperature, humidity, lux]

==============================================================================
OUTPUT
==============================================================================

data/processed/
    floor2_z1_hourly.csv
    floor2_z2_hourly.csv
    floor2_z4_hourly.csv
    floor5_z1_hourly.csv
    floor5_z4_hourly.csv
    floor5_z5_hourly.csv

Mỗi file có columns: [power, hour, day_of_week, temperature, humidity, lux,
                      raw_power]
Index: timestamp (hourly).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FILES = {
    "floor2": "2019Floor2.csv",
    "floor5": "2019Floor5.csv",
}

# ============================================================
# Config
# ============================================================


FEATURE_COLUMNS = [
    "power", "hour_sin", "hour_cos", "day_of_week", "temperature", "humidity", "lux",
]
CONTEXT_COLUMNS = ["hour_sin", "hour_cos", "day_of_week", "temperature", "humidity", "lux"]
# Zones cần giữ lại (tên theo pattern "z1", "z2", "z4")
SELECTED_ZONES = {
    "floor2": ["z1", "z2", "z4"],
    "floor5": ["z1", "z4", "z5"],
}


# ============================================================
# Helpers
# ============================================================

def _extract_zone_prefixes(columns: list[str]) -> dict[str, list[str]]:
    """Group columns by zone prefix (z1, z2, ...)."""
    zones: dict[str, list[str]] = {}
    for col in columns:
        if col == "Date":
            continue
        # pattern: z1_AC1(kW), z2_Light(kW), z2_S1(degC), etc.
        m = re.match(r"^(z\d+)_", col)
        if m:
            zone = m.group(1)
            zones.setdefault(zone, []).append(col)
    return zones


def _is_power_col(col: str) -> bool:
    """True nếu cột là power (kW)."""
    return "(kW)" in col


def _is_sensor_col(col: str, sensor_type: str) -> bool:
    """True nếu cột là sensor cụ thể: degC, RH%, hoặc lux."""
    return f"({sensor_type})" in col


def _extract_zone_features(
    df: pd.DataFrame,
    zone_cols: list[str],
) -> pd.DataFrame:
    """
    Từ 1 zone với nhiều cột (AC, Light, Plug, sensors), trích ra:
      - power = sum(all kW cols)
      - temperature = first (degC) col
      - humidity = first (RH%) col
      - lux = first (lux) col
    """
    power_cols = [c for c in zone_cols if _is_power_col(c)]
    temp_cols = [c for c in zone_cols if _is_sensor_col(c, "degC")]
    humid_cols = [c for c in zone_cols if _is_sensor_col(c, "RH%")]
    lux_cols = [c for c in zone_cols if _is_sensor_col(c, "lux")]

    out = pd.DataFrame(index=df.index)

    # Power: sum all kW cols (skipna=True để tolerant missing)
    if power_cols:
        out["power"] = df[power_cols].sum(axis=1, skipna=True)
    else:
        out["power"] = np.nan

    # Sensors: chỉ lấy cột đầu tiên (mỗi zone chỉ có 1 S1 sensor)
    out["temperature"] = df[temp_cols[0]] if temp_cols else np.nan
    out["humidity"] = df[humid_cols[0]] if humid_cols else np.nan
    out["lux"] = df[lux_cols[0]] if lux_cols else np.nan

    return out


def _resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample từ 1 phút -> 1 giờ bằng mean.
    Index của df phải là DatetimeIndex.
    """
    return df.resample("1h").mean()


def _derive_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive cyclic hour và day_of_week từ index."""
    df = df.copy()
    hour = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["day_of_week"] = df.index.dayofweek
    return df

def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize về [0, 1]."""
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return series * 0.0
    return (series - lo) / (hi - lo)


# ============================================================
# Main pipeline cho 1 floor
# ============================================================

def process_floor(
    floor_name: str,
    filename: str,
    selected_zones: list[str],
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Xử lý 1 floor file. Trả về dict {zone_id: processed_df}.
    """
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}")

    if verbose:
        print(f"\n{'='*72}")
        print(f"PROCESSING: {floor_name.upper()}  ({filename})")
        print(f"{'='*72}")

    # Đọc file, set Date làm index
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    if verbose:
        print(f"  Raw shape: {df.shape}")
        print(f"  Time range: {df.index.min()} -> {df.index.max()}")

    # Group columns theo zone
    zones_all = _extract_zone_prefixes(list(df.columns))
    if verbose:
        print(f"  Zones available: {sorted(zones_all.keys())}")
        print(f"  Zones selected: {selected_zones}")

    results: dict[str, pd.DataFrame] = {}

    for zone in selected_zones:
        if zone not in zones_all:
            print(f"  [WARN] Zone '{zone}' không tồn tại trong {floor_name}.")
            continue

        zone_cols = zones_all[zone]
        zone_df = _extract_zone_features(df, zone_cols)

        # Resample 1 phút -> 1 giờ
        zone_df = _resample_hourly(zone_df)

        # Derive temporal
        zone_df = _derive_temporal_features(zone_df)

        # Strict Causal Imputation: không dùng bfill()
        # 1. interpolate chỉ điền tiến (limit_direction="forward")
        zone_df[["temperature", "humidity", "lux"]] = (
            zone_df[["temperature", "humidity", "lux"]]
            .interpolate(method="linear", limit=3, limit_direction="forward")
            .ffill()
        )
        # 2. power chỉ forward-fill
        zone_df["power"] = zone_df["power"].ffill()

        # 3. Nếu những giờ đầu tiên của năm (00:00 ngày 01/01) bị NaN mà ffill không lấp được,
        # ta drop bỏ, tuyệt đối không lấy dữ liệu tương lai đắp ngược về quá khứ.
        zone_df = zone_df.dropna(subset=["power", "temperature", "humidity", "lux"])

        # Drop rows còn missing
        zone_df = zone_df.dropna(subset=["power", "temperature", "humidity", "lux"])

        # Normalize (giữ raw_power để báo cáo)
        zone_df["raw_power"] = zone_df["power"]
    
        # Reorder columns
        zone_df = zone_df[FEATURE_COLUMNS + ["raw_power"]]

        zone_id = f"{floor_name}_{zone}"
        results[zone_id] = zone_df

        if verbose:
            pct_missing = 100 * (1 - len(zone_df) / 8760)
            print(f"\n  Zone {zone}:")
            print(f"    Columns used (power): "
                  f"{sum(_is_power_col(c) for c in zone_cols)}")
            print(f"    Hourly shape: {zone_df.shape}")
            print(f"    Missing dropped: {pct_missing:.1f}%")
            print(f"    Power (raw): min={zone_df['raw_power'].min():.3f}  "
                  f"max={zone_df['raw_power'].max():.3f}  "
                  f"mean={zone_df['raw_power'].mean():.3f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, pd.DataFrame] = {}

    for floor_name, filename in FILES.items():
        selected = SELECTED_ZONES[floor_name]
        floor_results = process_floor(floor_name, filename, selected)
        all_results.update(floor_results)

    # Save
    print(f"\n{'='*72}")
    print("SAVING OUTPUTS")
    print(f"{'='*72}")

    for zone_id, df in all_results.items():
        out_path = PROCESSED_DIR / f"{zone_id}_hourly.csv"
        df.to_csv(out_path, index=True)
        print(f"  Saved: {out_path.name}  ({len(df)} rows)")

    print(f"\n{'='*72}")
    print(f"DONE — {len(all_results)} zone files saved to {PROCESSED_DIR}")
    print(f"{'='*72}")

    # Summary
    print(f"\nSUMMARY")
    print(f"{'-'*72}")
    print(f"  {'zone_id':20s}  {'n_hours':>8s}  "
          f"{'power_mean':>12s}  {'power_max':>10s}")
    print(f"  {'-'*20}  {'-'*8}  {'-'*12}  {'-'*10}")
    for zone_id, df in sorted(all_results.items()):
        print(f"  {zone_id:20s}  {len(df):8d}  "
              f"{df['raw_power'].mean():12.3f}  "
              f"{df['raw_power'].max():10.3f}")


if __name__ == "__main__":
    main()
# ============================================================
# Public API — compatible with CLEF pipeline
# ============================================================

ZONE_IDS = [
    "floor2_z1", "floor2_z2", "floor2_z4",
    "floor5_z1", "floor5_z4", "floor5_z5",
]


def load_zone_dataframe(zone_id: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load a processed zone file.

    Parameters
    ----------
    zone_id : str
        One of ZONE_IDS, e.g. 'floor2_z1'.
    verbose : bool

    Returns
    -------
    df : pd.DataFrame
        Index = timestamp (hourly).
        Columns = FEATURE_COLUMNS + ['raw_power'].
        Features already min-max normalized to [0, 1].
    """
    if zone_id not in ZONE_IDS:
        raise ValueError(f"Unknown zone_id '{zone_id}'. "
                         f"Available: {ZONE_IDS}")

    path = PROCESSED_DIR / f"{zone_id}_hourly.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Processed file not found: {path}. "
            f"Run `python src/data_prep_cubems.py` first."
        )

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()

    if verbose:
        print(f"[load_zone_dataframe] {zone_id}: shape={df.shape}  "
              f"time={df.index[0]} -> {df.index[-1]}")

    return df