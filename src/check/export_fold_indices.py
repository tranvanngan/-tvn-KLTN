import pandas as pd
import numpy as np

# Danh sách 6 file dữ liệu phân khu
zone_files = [
    "data/processed/floor2_z1_hourly.csv",
    "data/processed/floor2_z2_hourly.csv",
    "data/processed/floor2_z4_hourly.csv",
    "data/processed/floor5_z1_hourly.csv",
    "data/processed/floor5_z4_hourly.csv",
    "data/processed/floor5_z5_hourly.csv"
]

FOLD_RATIOS = [(0.45, 0.55, 0.70), (0.55, 0.70, 0.85), (0.70, 0.85, 1.00)]

for file_path in zone_files:
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"⚠️ Không tìm thấy file: {file_path} (Vui lòng kiểm tra lại đường dẫn)\n")
        continue

    N = len(df)
    zone_name = file_path.split("/")[-1].replace(".csv", "")
    
    # Tự động nhận diện cột thời gian (timestamp / time / datetime / cột 0)
    time_col = None
    for candidate in ['timestamp', 'time', 'datetime', 'date', 'Timestamp', 'Datetime', 'Time', 'Date']:
        if candidate in df.columns:
            time_col = candidate
            break
            
    if time_col is not None:
        time_series = df[time_col]
    else:
        # Nếu không có tên khớp, dùng trực tiếp cột đầu tiên
        time_series = df.iloc[:, 0]
        time_col = df.columns[0]
    
    print("=" * 105)
    print(f"📌 BÁO CÁO PHÂN CHIA FOLD CHO FILE: {zone_name} (Tổng số dòng N = {N:,})")
    print(f"   (Đã nhận diện cột thời gian: '{time_col}')")
    print("=" * 105)
    
    for fold_idx, (train_r, val_r, test_r) in enumerate(FOLD_RATIOS, 1):
        train_end = int(np.round(N * train_r))
        val_end = int(np.round(N * val_r))
        test_end = int(np.round(N * test_r)) if test_r < 1.0 else N
        
        # Mốc ngày giờ tương ứng
        t_train_start, t_train_end = time_series.iloc[0], time_series.iloc[train_end - 1]
        t_val_start,   t_val_end   = time_series.iloc[train_end], time_series.iloc[val_end - 1]
        t_test_start,  t_test_end  = time_series.iloc[val_end], time_series.iloc[test_end - 1]
        
        len_train = train_end
        len_val   = val_end - train_end
        len_test  = test_end - val_end
        
        print(f"--- FOLD {fold_idx} ---")
        print(f"  • Train [{0:4d} : {train_end:4d}) | {len_train:4d} dòng | Từ {t_train_start}  -->  {t_train_end}")
        print(f"  • Val   [{train_end:4d} : {val_end:4d}) | {len_val:4d} dòng | Từ {t_val_start}  -->  {t_val_end}")
        print(f"  • Test  [{val_end:4d} : {test_end:4d}) | {len_test:4d} dòng | Từ {t_test_start}  -->  {t_test_end}\n")