import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path

# ── 字型設定（支援中文）────────────────────────────────────────────────────
# Windows 使用微軟正黑體
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('./figures')
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 已儲存：figures/{name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Forward-fill 補值
# ══════════════════════════════════════════════════════════════════════════════
print("載入資料...")
df = pd.read_parquet('aqi_hourly_tidy.parquet')
df = df.sort_values(['station', 'date', 'hour']).reset_index(drop=True)

CRITICAL   = ['PM2.5', 'PM10', 'O3', 'CO', 'SO2', 'NO2']
AUXILIARY  = ['AMB_TEMP', 'RH', 'WIND_SPEED', 'WIND_DIREC',
              'WS_HR', 'WD_HR', 'RAINFALL', 'NO', 'NOx',
              'NMHC', 'THC', 'CH4', 'UVB']

# ── 3.1 補值前缺值比例（視覺化用）────────────────────────────────────────
cols_to_check = [c for c in CRITICAL + AUXILIARY if c in df.columns]
missing_before = df[cols_to_check].isna().mean() * 100

# ── 3.2 Forward-fill（最多 3 小時，按站分組）─────────────────────────────
print("Step 3：Forward-fill 補值中...")
df[cols_to_check] = (
    df.groupby('station')[cols_to_check]
    .transform(lambda s: s.ffill(limit=3))
)

missing_after = df[cols_to_check].isna().mean() * 100

# ── 3.3 視覺化：補值前後缺值比例比較 ─────────────────────────────────────
print("  繪製缺值比例比較圖...")
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(cols_to_check))
w = 0.35
ax.bar(x - w/2, missing_before, w, label='補值前', color='#E07B6A')
ax.bar(x + w/2, missing_after,  w, label='補值後', color='#5B9BD5')
ax.set_xticks(x)
ax.set_xticklabels(cols_to_check, rotation=45, ha='right')
ax.set_ylabel('缺值比例 (%)')
ax.set_title('Step 3：Forward-fill 補值前後缺值比例比較')
ax.legend()
ax.grid(axis='y', alpha=0.3)
save_fig('step3_missing_comparison')

# ── 3.4 視覺化：各站 × 年份 關鍵污染物缺值熱力圖 ─────────────────────────
print("  繪製缺值熱力圖...")
df['year'] = df['date'].dt.year
station_year_missing = (
    df.groupby(['station', 'year'])[CRITICAL]
    .apply(lambda g: g.isna().mean().mean() * 100)
    .reset_index(name='missing_pct')
    .pivot(index='station', columns='year', values='missing_pct')
)

fig, ax = plt.subplots(figsize=(14, 18))
sns.heatmap(
    station_year_missing, ax=ax,
    cmap='YlOrRd', annot=False,
    linewidths=0.3, linecolor='grey',
    cbar_kws={'label': '平均缺值率 (%)'}
)
ax.set_title('各測站 × 年份 關鍵污染物平均缺值率')
ax.set_xlabel('年份')
ax.set_ylabel('測站')
save_fig('step3_station_year_missing_heatmap')

print(f"Step 3 完成：{len(df):,} 列\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — 計算日 AQI 標籤
# ══════════════════════════════════════════════════════════════════════════════
print("Step 4：計算日 AQI 標籤...")

# ── 4.1 Taiwan MOE AQI 分段線性內插表 ────────────────────────────────────
# 格式：(污染物, averaging_hours, [(C_low, C_high, I_low, I_high), ...])
AQI_BREAKPOINTS = {
    'PM2.5': [
        (0.0,  12.4,   0,  50),
        (12.5, 35.4,  51, 100),
        (35.5, 54.4, 101, 150),
        (54.5, 150.4,151, 200),
        (150.5,250.4,201, 300),
        (250.5,325.4,301, 400),
        (325.5,500.4,401, 500),
    ],
    'PM10': [
        (0,   54,   0,  50),
        (55,  125,  51, 100),
        (126, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    'O3_8h': [
        (0.000, 0.054,  0,  50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085,101, 150),
        (0.086, 0.105,151, 200),
        (0.106, 0.200,201, 300),
    ],
    'O3_1h': [
        (0.101, 0.134,101, 150),
        (0.135, 0.204,151, 200),
        (0.205, 0.404,201, 300),
        (0.405, 0.504,301, 400),
        (0.505, 0.604,401, 500),
    ],
    'CO': [
        (0.0,  4.4,   0,  50),
        (4.5,  9.4,  51, 100),
        (9.5,  12.4,101, 150),
        (12.5, 15.4,151, 200),
        (15.5, 30.4,201, 300),
        (30.5, 40.4,301, 400),
        (40.5, 50.4,401, 500),
    ],
    'SO2': [
        (0,    20,   0,  50),
        (21,   75,  51, 100),
        (76,  185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805,1004, 401, 500),
    ],
    'NO2': [
        (0,    53,   0,  50),
        (54,  100,  51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650,1249, 201, 300),
        (1250,1649,301, 400),
        (1650,2049,401, 500),
    ],
}

def linear_interp(C, breakpoints):
    """分段線性內插計算子指標"""
    for (C_lo, C_hi, I_lo, I_hi) in breakpoints:
        if C_lo <= C <= C_hi:
            return I_lo + (I_hi - I_lo) / (C_hi - C_lo) * (C - C_lo)
    return np.nan

def sub_index(C, pollutant):
    if pd.isna(C):
        return np.nan
    bp = AQI_BREAKPOINTS.get(pollutant)
    if bp is None:
        return np.nan
    return linear_interp(C, bp)

# ── 4.2 計算各污染物的 rolling average（按站分組）────────────────────────
print("  計算 rolling averages...")
df = df.sort_values(['station', 'date', 'hour']).reset_index(drop=True)

grp = df.groupby('station')

# PM2.5, PM10 → 24hr 平均
df['PM2.5_24h'] = grp['PM2.5'].transform(
    lambda s: s.rolling(24, min_periods=18).mean())
df['PM10_24h']  = grp['PM10'].transform(
    lambda s: s.rolling(24, min_periods=18).mean())

# O3 → 8hr 平均 + 1hr（原始值）
df['O3_8h'] = grp['O3'].transform(
    lambda s: s.rolling(8, min_periods=6).mean())
df['O3_1h'] = df['O3']

# CO → 8hr 平均
df['CO_8h'] = grp['CO'].transform(
    lambda s: s.rolling(8, min_periods=6).mean())

# SO2, NO2 → 1hr（原始值）
df['SO2_1h'] = df['SO2']
df['NO2_1h'] = df['NO2']

# ── 4.3 計算各污染物子指標 ────────────────────────────────────────────────
print("  計算 AQI 子指標...")
df['SI_PM25'] = df['PM2.5_24h'].apply(lambda x: sub_index(x, 'PM2.5'))
df['SI_PM10'] = df['PM10_24h'].apply( lambda x: sub_index(x, 'PM10'))
df['SI_O3_8h']= df['O3_8h'].apply(   lambda x: sub_index(x, 'O3_8h'))
df['SI_O3_1h']= df['O3_1h'].apply(   lambda x: sub_index(x, 'O3_1h'))
df['SI_CO']   = df['CO_8h'].apply(   lambda x: sub_index(x, 'CO'))
df['SI_SO2']  = df['SO2_1h'].apply(  lambda x: sub_index(x, 'SO2'))
df['SI_NO2']  = df['NO2_1h'].apply(  lambda x: sub_index(x, 'NO2'))

# O3：取 8hr 與 1hr 子指標的最大值（附註 1）
df['SI_O3'] = df[['SI_O3_8h', 'SI_O3_1h']].max(axis=1)

# ── 4.4 每小時 AQI = 各子指標最大值 ──────────────────────────────────────
si_cols = ['SI_PM25', 'SI_PM10', 'SI_O3', 'SI_CO', 'SI_SO2', 'SI_NO2']
df['AQI_hourly'] = df[si_cols].max(axis=1)

# ── 4.5 聚合為日 AQI（每站每日取最大值）──────────────────────────────────
print("  聚合為日 AQI...")
daily = (
    df.groupby(['station', 'date'])
    .agg(
        AQI          = ('AQI_hourly', 'max'),
        SI_PM25_max  = ('SI_PM25',    'max'),
        SI_PM10_max  = ('SI_PM10',    'max'),
        SI_O3_max    = ('SI_O3',      'max'),
        SI_CO_max    = ('SI_CO',      'max'),
        SI_SO2_max   = ('SI_SO2',     'max'),
        SI_NO2_max   = ('SI_NO2',     'max'),
        # 日均氣象特徵
        AMB_TEMP_mean= ('AMB_TEMP',   'mean'),
        RH_mean      = ('RH',         'mean'),
        WIND_SPEED_mean=('WIND_SPEED','mean'),
        WIND_DIREC_mean=('WIND_DIREC','mean'),
        RAINFALL_sum = ('RAINFALL',   'sum'),
    )
    .reset_index()
)

# ── 4.6 AQI 分類標籤 ──────────────────────────────────────────────────────
def classify_aqi(aqi):
    if pd.isna(aqi): return np.nan
    if aqi <= 50:    return 0  # Good
    if aqi <= 100:   return 1  # Moderate
    if aqi <= 150:   return 2  # Unhealthy for Sensitive
    if aqi <= 200:   return 3  # Unhealthy
    if aqi <= 300:   return 4  # Very Unhealthy
    return 5                   # Hazardous

LABEL_NAMES = {
    0: 'Good',
    1: 'Moderate',
    2: 'Unhealthy (Sensitive)',
    3: 'Unhealthy',
    4: 'Very Unhealthy',
    5: 'Hazardous'
}
LABEL_COLORS = ['#00E400','#FFFF00','#FF7E00','#FF0000','#8F3F97','#7E0023']

daily['AQI_label'] = daily['AQI'].apply(classify_aqi)
daily['AQI_label'] = daily['AQI_label'].astype('Int64')  # 允許 NaN 的整數型別
daily['year']  = daily['date'].dt.year
daily['month'] = daily['date'].dt.month

print(f"  日 AQI 資料集：{len(daily):,} 筆")
print(f"  AQI 標籤分布：\n{daily['AQI_label'].value_counts().sort_index()}\n")

# ── 4.7 視覺化：AQI 等級分布（全台）─────────────────────────────────────
print("  繪製 AQI 等級分布圖...")
label_counts = daily['AQI_label'].value_counts().sort_index()
labels_exist = [LABEL_NAMES[int(i)] for i in label_counts.index]
colors_exist = [LABEL_COLORS[int(i)] for i in label_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 長條圖
axes[0].bar(labels_exist, label_counts.values, color=colors_exist, edgecolor='grey')
axes[0].set_title('全台日 AQI 等級分布（2016–2025）')
axes[0].set_xlabel('AQI 等級')
axes[0].set_ylabel('天數')
axes[0].tick_params(axis='x', rotation=30)
for i, v in enumerate(label_counts.values):
    axes[0].text(i, v + 200, f'{v:,}', ha='center', fontsize=9)

# 圓餅圖
axes[1].pie(
    label_counts.values,
    labels=labels_exist,
    colors=colors_exist,
    autopct='%1.1f%%',
    startangle=140
)
axes[1].set_title('日 AQI 等級比例')
plt.tight_layout()
save_fig('step4_aqi_label_distribution')

# ── 4.8 視覺化：月份 × AQI 等級熱力圖（季節性）──────────────────────────
print("  繪製月份 × AQI 等級熱力圖...")
month_label = (
    daily.groupby(['month', 'AQI_label'])
    .size().reset_index(name='count')
    .pivot(index='AQI_label', columns='month', values='count')
    .fillna(0)
)
month_label.index = [LABEL_NAMES[i] for i in month_label.index]
month_label.columns = [f'{m}月' for m in month_label.columns]

fig, ax = plt.subplots(figsize=(14, 5))
sns.heatmap(month_label, ax=ax, cmap='YlOrRd',
            annot=True, fmt='.0f', linewidths=0.3,
            cbar_kws={'label': '天數'})
ax.set_title('月份 × AQI 等級分布（季節性分析）')
ax.set_xlabel('月份')
ax.set_ylabel('AQI 等級')
save_fig('step4_monthly_aqi_heatmap')

# ── 4.9 視覺化：年份趨勢（各等級天數）────────────────────────────────────
print("  繪製年份趨勢圖...")
year_label = (
    daily.groupby(['year', 'AQI_label'])
    .size().reset_index(name='count')
    .pivot(index='year', columns='AQI_label', values='count')
    .fillna(0)
)
year_label.columns = [LABEL_NAMES[i] for i in year_label.columns]

fig, ax = plt.subplots(figsize=(12, 5))
year_label.plot(kind='bar', stacked=True,
                color=LABEL_COLORS[:len(year_label.columns)],
                ax=ax, edgecolor='white', linewidth=0.5)
ax.set_title('年份 × AQI 等級天數趨勢（2016–2025）')
ax.set_xlabel('年份')
ax.set_ylabel('天數（站 × 日）')
ax.legend(title='AQI 等級', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.tick_params(axis='x', rotation=45)
save_fig('step4_yearly_aqi_trend')

# ── 4.10 視覺化：AQI 分布箱型圖（各主要污染物子指標）────────────────────
print("  繪製子指標箱型圖...")
si_plot_cols = ['SI_PM25_max','SI_PM10_max','SI_O3_max',
                'SI_CO_max','SI_SO2_max','SI_NO2_max']
si_plot_cols = [c for c in si_plot_cols if c in daily.columns]
si_labels    = ['PM2.5','PM10','O3','CO','SO2','NO2']

fig, ax = plt.subplots(figsize=(12, 5))
daily[si_plot_cols].boxplot(ax=ax, patch_artist=True,
    boxprops=dict(facecolor='#5B9BD5', color='navy'),
    medianprops=dict(color='red', linewidth=2))
ax.set_xticklabels(si_labels[:len(si_plot_cols)])
ax.set_title('各污染物 AQI 子指標分布')
ax.set_ylabel('子指標值')
ax.axhline(50,  color='green',  linestyle='--', alpha=0.5, label='Good/Moderate 邊界')
ax.axhline(100, color='orange', linestyle='--', alpha=0.5, label='Moderate/敏感 邊界')
ax.axhline(150, color='red',    linestyle='--', alpha=0.5, label='敏感/Unhealthy 邊界')
ax.legend(fontsize=8)
save_fig('step4_subindex_boxplot')

# ── 4.11 儲存日 AQI 資料集 ────────────────────────────────────────────────
daily.to_parquet('aqi_daily.parquet', index=False)
print(f"\n儲存完成：aqi_daily.parquet")
print(f"檔案大小：{Path('aqi_daily.parquet').stat().st_size / 1024**2:.1f} MB")
print(f"\n=== Step 3–4 全部完成 ===")
print(f"日 AQI 資料集：{len(daily):,} 筆 × {len(daily.columns)} 欄")
print(daily[['station','date','AQI','AQI_label']].head(10).to_string())