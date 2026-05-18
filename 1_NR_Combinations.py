import pandas as pd
import glob
import os

# ── 1. 載入所有檔案（支援 csv / xls / xlsx）──────────────────────────────
def load_file(path):
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == '.csv':
            return pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
        elif ext in ('.xls', '.xlsx'):
            return pd.read_excel(path, engine='xlrd' if ext == '.xls' else 'openpyxl')
    except Exception as e:
        print(f"[SKIP] {path} — {e}")
    return None

all_files = glob.glob('./data/**/*.*', recursive=True)
all_files = [f for f in all_files if os.path.splitext(f)[1].lower() in ('.csv', '.xls', '.xlsx')]

df_raw = pd.concat(
    [df for f in all_files if (df := load_file(f)) is not None],
    ignore_index=True
)

# ── 2. 只保留 RAIN_COND / RAINFALL / PH_RAIN 三個測項 ────────────────────
RAIN_ITEMS = ['RAIN_COND', 'RAINFALL', 'PH_RAIN']
df_rain = df_raw[df_raw['測項'].isin(RAIN_ITEMS)].copy()

# ── 3. Melt 24 小時欄位 ───────────────────────────────────────────────────
hour_cols = [c for c in df_rain.columns if str(c).strip() in [f'{h:02d}' for h in range(24)]]
df_long = df_rain.melt(
    id_vars=['測站', '日期', '測項'],
    value_vars=hour_cols,
    var_name='hour',
    value_name='raw_value'
)

# ── 4. 分類函數：NaN / NR / Val ───────────────────────────────────────────
INVALID_MARKERS = {'#', '*', 'x', 'A', ''}

def classify(v):
    if pd.isna(v):
        return 'NaN'
    s = str(v).strip()
    if s in INVALID_MARKERS or s == 'nan':
        return 'NaN'
    if s == 'NR':
        return 'NR'
    try:
        float(s)
        return 'Val'
    except ValueError:
        return 'NaN'  # 未知標記也歸 NaN

df_long['category'] = df_long['raw_value'].apply(classify)

# ── 5. Pivot：每個 (測站, 日期, hour) 得到三欄的 category ─────────────────
df_pivot = df_long.pivot_table(
    index=['測站', '日期', 'hour'],
    columns='測項',
    values='category',
    aggfunc='first'
).reset_index()
df_pivot.columns.name = None

# 確保三欄都存在（部分站可能完全缺某測項）
for col in RAIN_ITEMS:
    if col not in df_pivot.columns:
        df_pivot[col] = 'NaN'

# ── 6. 組合欄位 ───────────────────────────────────────────────────────────
df_pivot['combo'] = (
    df_pivot['RAIN_COND'].fillna('NaN') + ' | ' +
    df_pivot['RAINFALL'].fillna('NaN') + ' | ' +
    df_pivot['PH_RAIN'].fillna('NaN')
)

# ── 7. 統計與排序 ─────────────────────────────────────────────────────────
total = len(df_pivot)
combo_counts = df_pivot['combo'].value_counts()
combo_pct    = (combo_counts / total * 100).round(3)

summary = pd.DataFrame({
    'count': combo_counts,
    'pct_%': combo_pct
})

print(f"Total station-hour records (rain vars): {total:,}")
print(f"Unique combinations: {len(summary)}\n")
print(summary.to_string())

# ── 8. 輸出 CSV ───────────────────────────────────────────────────────────
summary.to_csv('nr_combo_audit.csv')
print("\nSaved to nr_combo_audit.csv")