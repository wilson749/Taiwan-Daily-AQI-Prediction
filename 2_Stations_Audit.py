import pandas as pd
import numpy as np
import os
import glob
import re

# ── 1. 遞迴掃描所有檔案，建立索引 ────────────────────────────────────────
def extract_station_year(filepath):
    """
    從路徑或檔名中萃取測站名與西元年份。
    支援兩種命名格式：
      - 測站名_西元年.csv         (e.g. 富貴角_2025.csv)
      - 民國年測站名站_發布日期.xls (e.g. 103年小港站_20170317.xls)
    """
    filename = os.path.splitext(os.path.basename(filepath))[0]

    # 格式 A：測站名_西元年 (e.g. 富貴角_2025)
    match_a = re.search(r'_(\d{4})$', filename)
    if match_a:
        year = int(match_a.group(1))
        station = filename[:match_a.start()]
        if 2000 <= year <= 2030:
            return station, year

    # 格式 B：民國年+測站名+站 (e.g. 103年小港站_20170317)
    match_b = re.match(r'^(\d{2,3})年(.+?)站', filename)
    if match_b:
        minguo = int(match_b.group(1))
        station = match_b.group(2)
        year = minguo + 1911
        if 2000 <= year <= 2030:
            return station, year

    return None, None


def build_file_index(data_root='./data'):
    records = []
    exts = ('.csv', '.xls', '.xlsx')
    for path in glob.glob(os.path.join(data_root, '**', '*.*'), recursive=True):
        if os.path.splitext(path)[1].lower() not in exts:
            continue
        station, year = extract_station_year(path)
        if station and year:
            records.append({'station': station, 'year': year, 'filepath': path})
        else:
            print(f"[WARN] 無法解析站名/年份：{path}")

    index = pd.DataFrame(records)
    print(f"索引完成：{len(index)} 個檔案，"
          f"{index['station'].nunique()} 站，"
          f"{index['year'].nunique()} 年份\n")
    return index


# ── 2. 讀取單一檔案，回傳該檔包含的測項清單 ──────────────────────────────
def get_items_in_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.csv':
            df = pd.read_csv(filepath, encoding='utf-8-sig',
                             usecols=['測項'], low_memory=False)
        else:
            df = pd.read_excel(filepath,
                               engine='xlrd' if ext == '.xls' else 'openpyxl',
                               usecols=['測項'])
        return set(df['測項'].dropna().unique())
    except Exception as e:
        print(f"[SKIP] {filepath} — {e}")
        return set()


# ── 3. 對每個站年組合執行 coverage 檢查 ───────────────────────────────────
CRITICAL = {'PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2'}

def run_coverage_audit(index):
    rows = []
    total = len(index)
    for i, row in index.iterrows():
        if i % 50 == 0:
            print(f"  進度：{i}/{total}...")
        items = get_items_in_file(row['filepath'])
        entry = {
            'station': row['station'],
            'year':    row['year'],
        }
        # 六項關鍵污染物各自是否存在
        for pollutant in sorted(CRITICAL):
            entry[pollutant] = pollutant in items
        # 是否六項齊全
        entry['all_critical'] = all(entry[p] for p in CRITICAL)
        rows.append(entry)

    audit = pd.DataFrame(rows).sort_values(['station', 'year']).reset_index(drop=True)
    return audit


# ── 4. 彙總統計 ───────────────────────────────────────────────────────────
def print_summary(audit):
    total        = len(audit)
    complete     = audit['all_critical'].sum()
    incomplete   = total - complete
    pct_complete = complete / total * 100

    print(f"\n{'='*55}")
    print(f"站年組合總數：     {total}")
    print(f"六項齊全（可用）：  {complete}  ({pct_complete:.1f}%)")
    print(f"缺少至少一項：     {incomplete}  ({100-pct_complete:.1f}%)")
    print(f"{'='*55}\n")

    # 各污染物缺失比例
    print("各關鍵污染物覆蓋率：")
    for p in sorted(CRITICAL):
        covered = audit[p].sum()
        print(f"  {p:<8} {covered}/{total}  ({covered/total*100:.1f}%)")

    # 哪些站從未達到六項齊全
    never_complete = (
        audit.groupby('station')['all_critical']
        .any()
        .pipe(lambda s: s[~s].index.tolist())
    )
    if never_complete:
        print(f"\n從未六項齊全的測站（{len(never_complete)} 站）：")
        for s in never_complete:
            print(f"  {s}")

    # 年份維度：各年可用站數
    print("\n各年可用站數（六項齊全）：")
    yr = audit.groupby('year')['all_critical'].sum().sort_index()
    for year, count in yr.items():
        print(f"  {year}：{count} 站")


# ── 5. 主流程 ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # 修改為你的資料根目錄
    DATA_ROOT = './data'

    print("Step 1：建立檔案索引...")
    index = build_file_index(DATA_ROOT)

    print("Step 2：掃描各檔案測項...")
    audit = run_coverage_audit(index)

    print("Step 3：彙總結果...")
    print_summary(audit)

    # 輸出完整 audit 表
    audit.to_csv('coverage_audit.csv', index=False)
    print("\n完整結果已儲存至 coverage_audit.csv")

    # 輸出可用站年組合清單
    usable = audit[audit['all_critical']].copy()
    usable.to_csv('usable_station_years.csv', index=False)
    print(f"可用站年組合已儲存至 usable_station_years.csv（共 {len(usable)} 筆）")