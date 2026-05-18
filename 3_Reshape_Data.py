import pandas as pd
import numpy as np
import os
import re
import glob

# ── Step 1: Recursive file index with Minguo year conversion ────────────────────────────────────
def extract_station_year(filepath):
    filename = os.path.splitext(os.path.basename(filepath))[0]
    match_a = re.search(r'_(\d{4})$', filename)
    if match_a:
        year = int(match_a.group(1))
        station = filename[:match_a.start()]
        if 2000 <= year <= 2030:
            return station, year
    match_b = re.match(r'^(\d{2,3})年(.+?)站', filename)
    if match_b:
        minguo = int(match_b.group(1))
        station = match_b.group(2)
        year = minguo + 1911
        if 2000 <= year <= 2030:
            return station, year
    return None, None

DATA_ROOT = './data'  # 資料根目錄

records = []
exts = ('.csv', '.xls', '.xlsx')
for path in glob.glob(os.path.join(DATA_ROOT, '**', '*.*'), recursive=True):
    if os.path.splitext(path)[1].lower() not in exts:
        continue
    station, year = extract_station_year(path)
    if station and year:
        records.append({'station': station, 'year': year, 'filepath': path})

file_index = pd.DataFrame(records)

# 讀取可用站年清單，與檔案索引合併取得 filepath
usable = pd.read_csv('usable_station_years.csv')
usable = usable[usable['all_critical']].merge(
    file_index[['station', 'year', 'filepath']],
    on=['station', 'year'],
    how='inner'
).reset_index(drop=True)
print(f"載入可用站年組合：{len(usable)} 筆")
print(f"年份分布：\n{usable['year'].value_counts().sort_index()}")

# 在 build_file_index 之後加入這行確認
print(file_index.groupby(['year', file_index['filepath'].str.endswith('.xls')])['station'].count())

# ── 2. 讀取單一檔案 ───────────────────────────────────────────────────────
def load_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath, encoding='utf-8-sig',
                             header=0, low_memory=False)
        elif ext == '.xls':
            df = pd.read_excel(filepath, engine='xlrd', header=0)
        elif ext == '.xlsx':
            df = pd.read_excel(filepath, engine='openpyxl', header=0)
        else:
            return None

        # 移除 Unnamed 尾端欄位
        df.drop(columns=[c for c in df.columns if 'Unnamed' in str(c)],
                inplace=True)

        # 統一欄位順序：XLS 是 日期/測站/測項，CSV 是 測站/日期/測項
        if '日期' in df.columns and df.columns[0] == '日期':
            cols = df.columns.tolist()
            for c in ['日期', '測站', '測項']:
                cols.remove(c)
            df = df[['測站', '日期', '測項'] + cols]

        # 統一小時欄位名稱為零填充字串
        rename_map = {}
        for col in df.columns:
            s = str(col).strip()
            if s.isdigit() and 0 <= int(s) <= 23:
                rename_map[col] = f'{int(s):02d}'
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # 移除重複欄位
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    except Exception as e:
        print(f"[SKIP] {filepath} — {e}")
        return None

# ── 3. 批次載入並合併 ─────────────────────────────────────────────────────
dfs = []
total = len(usable)

for i, (_, row) in enumerate(usable.iterrows()):
    if i % (0.2 * total) == 0:
        print(f"  載入進度：{i}/{total}...")
    df = load_file(row['filepath'])
    if df is not None:
        df['_station'] = row['station']
        df['_year']    = row['year']
        dfs.append(df)
    else:
        print(f"  [SKIP] {row['filepath']}")

df_raw = pd.concat(dfs, ignore_index=True)
print(f"\n合併完成：{len(df_raw):,} 原始列")
print(f"欄位：{list(df_raw.columns)}\n")

# 以檔案索引的站名為準，確保站名一致
df_raw['測站'] = df_raw['_station']
df_raw.drop(columns=['_station', '_year'], inplace=True)

# ── 4. Melt：wide → long ──────────────────────────────────────────────────
hour_cols = [c for c in df_raw.columns
             if str(c).strip() in {f'{h:02d}' for h in range(24)}]

print(f"偵測到小時欄位數：{len(hour_cols)}（預期 24）")
assert len(hour_cols) == 24, "小時欄位數異常，請檢查原始資料"

df_long = df_raw.melt(
    id_vars=['測站', '日期', '測項'],
    value_vars=hour_cols,
    var_name='hour',
    value_name='raw_value'
)

# 統一日期格式（取前10字元 YYYY/MM/DD，去除時間部分）
df_long['日期'] = df_long['日期'].astype(str).str.strip().str[:10]

print(f"Melt 完成：{len(df_long):,} 列\n")
del df_raw


# ── 5. 排除 RAIN_COND、PH_RAIN ────────────────────────────────────────────
df_long = df_long[~df_long['測項'].isin(['RAIN_COND', 'PH_RAIN'])]


# ── 6. 清洗 raw_value ─────────────────────────────────────────────────────
INVALID_MARKERS = {'#', '*', 'x', 'A', ''}

def clean_value(v, item):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s in INVALID_MARKERS or s.lower() == 'nan':
        return np.nan
    # RAINFALL 的 NR = 0（無降雨）
    if s == 'NR':
        return 0.0 if item == 'RAINFALL' else np.nan
    try:
        val = float(s)
        # 負值視為異常（污染物濃度不可為負）
        if item not in ('AMB_TEMP',) and val < 0:
            return np.nan
        return val
    except ValueError:
        return np.nan

# 日期診斷（清洗前）
print("日期診斷：")
print(df_long['日期'].sort_values().iloc[:3].tolist())
print(df_long['日期'].sort_values().iloc[-3:].tolist())

print("清洗數值中（需要幾分鐘）...")

# 先將無效標記統一替換為 NaN（向量化，不用 list comprehension）
invalid = {'#', '*', 'x', 'A', '', 'nan'}
df_long['raw_value'] = df_long['raw_value'].astype(str).str.strip()
df_long.loc[df_long['raw_value'].isin(invalid), 'raw_value'] = np.nan

# RAINFALL 的 NR 填 0，其餘 NR 填 NaN
rainfall_mask = df_long['測項'] == 'RAINFALL'
df_long.loc[rainfall_mask & (df_long['raw_value'] == 'NR'), 'raw_value'] = '0'
df_long.loc[~rainfall_mask & (df_long['raw_value'] == 'NR'), 'raw_value'] = np.nan

# 轉為數值
df_long['value'] = pd.to_numeric(df_long['raw_value'], errors='coerce')

# 負值處理（AMB_TEMP 允許負值）
non_temp = df_long['測項'] != 'AMB_TEMP'
df_long.loc[non_temp & (df_long['value'] < 0), 'value'] = np.nan

df_long.drop(columns=['raw_value'], inplace=True)
print(f"清洗完成\n")


# ── 7. Pivot：long → tidy（每列 = 一站一日一小時）────────────────────────
print("df_long 日期樣本（排序後頭尾）：")
dates = pd.to_datetime(df_long['日期'], errors='coerce')
print(dates.sort_values().dropna().iloc[[0, 1, -2, -1]])
print(f"\n無法解析的日期數量：{dates.isna().sum():,}")
print(f"\n日期原始值樣本（前5、後5）：")
print(df_long['日期'].sort_values().iloc[:5].tolist())
print(df_long['日期'].sort_values().iloc[-5:].tolist())

print("Pivot 中...")
df_tidy = df_long.pivot_table(
    index=['測站', '日期', 'hour'],
    columns='測項',
    values='value',
    aggfunc='first'
).reset_index()
df_tidy.columns.name = None

# 統一欄位名稱
df_tidy.rename(columns={'測站': 'station', '日期': 'date'}, inplace=True)

# 解析日期欄位（處理兩種格式混用）
df_tidy['date'] = pd.to_datetime(
    df_tidy['date'].astype(str).str.strip().str[:10],
    format='%Y/%m/%d',
    errors='coerce'
)

print(f"Pivot 完成：{len(df_tidy):,} 列 × {len(df_tidy.columns)} 欄")
print(f"欄位清單：{list(df_tidy.columns)}\n")


# ── 8. 確認輸出結構 ───────────────────────────────────────────────────────
print("=== 資料樣貌確認 ===")
print(f"日期範圍：{df_tidy['date'].min()} ~ {df_tidy['date'].max()}")
print(f"測站數：{df_tidy['station'].nunique()}")
print(f"每站預期列數（10年×365天×24小時）：{10*365*24:,}")
print(f"\n缺值比例（關鍵污染物）：")
for col in ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']:
    if col in df_tidy.columns:
        pct = df_tidy[col].isna().mean() * 100
        print(f"  {col:<8} {pct:.2f}%")

# ── 9. 儲存 ──────────────────────────────────────────────────────────────
out_path = 'aqi_hourly_tidy.parquet'
df_tidy.to_parquet(out_path, index=False)
print(f"\n儲存完成：{out_path}")
print(f"檔案大小：{os.path.getsize(out_path) / 1024**2:.1f} MB")