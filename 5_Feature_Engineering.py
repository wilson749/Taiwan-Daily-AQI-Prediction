import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('./figures')
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 已儲存：figures/{name}.png")

# ══════════════════════════════════════════════════════════════════════════════
# 載入資料
# ══════════════════════════════════════════════════════════════════════════════
print("載入 aqi_daily.parquet...")
df = pd.read_parquet('aqi_daily.parquet')
df = df.sort_values(['station', 'date']).reset_index(drop=True)
print(f"載入完成：{len(df):,} 筆 × {len(df.columns)} 欄\n")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — 特徵工程
# ══════════════════════════════════════════════════════════════════════════════
print("Step 5：特徵工程...")

# ── 5.1 時間特徵 ──────────────────────────────────────────────────────────
print("  5.1 時間特徵...")
df['month']       = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek   # 0=Monday, 6=Sunday
df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
df['season']      = df['month'].map({
    12: 0, 1: 0, 2: 0,   # 冬
     3: 1, 4: 1, 5: 1,   # 春
     6: 2, 7: 2, 8: 2,   # 夏
     9: 3,10: 3,11: 3    # 秋
})

# ── 5.2 風向 sin/cos 編碼 ─────────────────────────────────────────────────
print("  5.2 風向 encoding...")
df['wind_sin'] = np.sin(np.deg2rad(df['WIND_DIREC_mean']))
df['wind_cos'] = np.cos(np.deg2rad(df['WIND_DIREC_mean']))

# ── 5.3 地區分類 ──────────────────────────────────────────────────────────
print("  5.3 地區分類...")
REGION_MAP = {
    # 北部：臺北市、新北市、基隆市、桃園市、新竹市、新竹縣、宜蘭縣
    '基隆': '北部', '士林': '北部', '中山': '北部', '松山': '北部',
    '萬華': '北部', '古亭': '北部', '大同': '北部', '板橋': '北部',
    '新莊': '北部', '永和': '北部', '土城': '北部', '汐止': '北部',
    '林口': '北部', '淡水': '北部', '萬里': '北部', '富貴角': '北部',
    '陽明': '北部', '新店': '北部', '三重': '北部',
    '桃園': '北部', '中壢': '北部', '平鎮': '北部', '龍潭': '北部',
    '大園': '北部', '觀音': '北部', '湖口': '北部',
    '新竹': '北部', '竹東': '北部',
    '宜蘭': '北部', '冬山': '北部',
    # 中部：臺中市、苗栗縣、彰化縣、南投縣、雲林縣
    '頭份': '中部', '苗栗': '中部', '三義': '中部',
    '豐原': '中部', '沙鹿': '中部', '西屯': '中部',
    '忠明': '中部', '大里': '中部',
    '彰化': '中部', '線西': '中部', '二林': '中部',
    '南投': '中部', '埔里': '中部', '竹山': '中部',
    '崙背': '中部', '臺西': '中部', '麥寮': '中部', '斗六': '中部',
    # 南部：高雄市、臺南市、嘉義市、嘉義縣、屏東縣、澎湖縣
    '朴子': '南部', '嘉義': '南部', '新港': '南部',
    '新營': '南部', '善化': '南部', '安南': '南部', '臺南': '南部',
    '左營': '南部', '楠梓': '南部', '橋頭': '南部', '仁武': '南部',
    '大寮': '南部', '林園': '南部', '鳳山': '南部', '前鎮': '南部',
    '小港': '南部', '前金': '南部', '菜寮': '南部', '美濃': '南部',
    '潮州': '南部', '屏東': '南部', '恆春': '南部',
    '馬公': '南部',
    # 東部：花蓮縣、臺東縣
    '花蓮': '東部', '臺東': '東部', '關山': '東部',
    # 離島：金門、馬祖
    '金門': '離島', '馬祖': '離島',
}
df['region'] = df['station'].map(REGION_MAP).fillna('其他')
df['region_code'] = df['region'].map(
    {'北部': 0, '中部': 1, '南部': 2, '東部': 3, '離島': 4, '其他': 5}
)

# ── 5.4 Lag 特徵（前一日 AQI）────────────────────────────────────────────
print("  5.4 Lag 特徵...")
df['AQI_lag1'] = df.groupby('station')['AQI'].shift(1)
df['AQI_lag2'] = df.groupby('station')['AQI'].shift(2)
df['AQI_lag3'] = df.groupby('station')['AQI'].shift(3)
df['AQI_label_lag1'] = df.groupby('station')['AQI_label'].shift(1)

# ── 5.5 7日滾動平均（AQI 趨勢）───────────────────────────────────────────
print("  5.5 AQI 7日滾動平均...")
df['AQI_7d_mean'] = df.groupby('station')['AQI'].transform(
    lambda s: s.rolling(7, min_periods=4).mean()
)

# ── 5.6 移除 lag 產生的前幾列 NaN ────────────────────────────────────────
df_model = df.dropna(subset=['AQI_lag1', 'AQI_label']).copy()
print(f"  移除 lag NaN 後：{len(df_model):,} 筆\n")

# ══════════════════════════════════════════════════════════════════════════════
# 視覺化
# ══════════════════════════════════════════════════════════════════════════════

LABEL_NAMES  = {0:'Good', 1:'Moderate', 2:'Unhealthy(Sen.)',
                3:'Unhealthy', 4:'Very Unhealthy', 5:'Hazardous'}
LABEL_COLORS = ['#00E400','#FFFF00','#FF7E00','#FF0000','#8F3F97','#7E0023']

# ── 5.7 各地區 AQI 等級分布 ──────────────────────────────────────────────
print("  繪製各地區 AQI 等級分布...")
region_order = ['北部', '中部', '南部', '東部', '離島']
region_label = (
    df_model[df_model['region'].isin(region_order)]
    .groupby(['region', 'AQI_label'])
    .size().reset_index(name='count')
    .pivot(index='region', columns='AQI_label', values='count')
    .fillna(0)
    .reindex(region_order)
)
region_label.columns = [LABEL_NAMES[int(c)] for c in region_label.columns]

fig, ax = plt.subplots(figsize=(12, 5))
region_label.plot(kind='bar', stacked=True, ax=ax,
                  color=LABEL_COLORS[:len(region_label.columns)],
                  edgecolor='white', linewidth=0.5)
ax.set_title('各地區 AQI 等級分布比較')
ax.set_xlabel('地區')
ax.set_ylabel('站日數')
ax.tick_params(axis='x', rotation=0)
ax.legend(title='AQI 等級', bbox_to_anchor=(1.01, 1), loc='upper left')
save_fig('step5_region_aqi_distribution')

# ── 5.8 月份 × 地區 AQI 均值熱力圖 ──────────────────────────────────────
print("  繪製月份 × 地區 AQI 均值熱力圖...")
pivot = (
    df_model[df_model['region'].isin(region_order)]
    .groupby(['region', 'month'])['AQI']
    .mean()
    .reset_index()
    .pivot(index='region', columns='month', values='AQI')
    .reindex(region_order)
)
pivot.columns = [f'{m}月' for m in pivot.columns]

fig, ax = plt.subplots(figsize=(14, 4))
sns.heatmap(pivot, ax=ax, cmap='YlOrRd', annot=True, fmt='.0f',
            linewidths=0.3, cbar_kws={'label': '平均 AQI'})
ax.set_title('各地區月份平均 AQI（季節性 × 地區）')
ax.set_xlabel('月份')
ax.set_ylabel('地區')
save_fig('step5_region_month_aqi_heatmap')

# ── 5.9 前一日 AQI vs 當日 AQI 散佈圖 ────────────────────────────────────
print("  繪製 Lag1 AQI 散佈圖...")
sample = df_model.sample(min(5000, len(df_model)), random_state=42)
fig, ax = plt.subplots(figsize=(7, 6))
scatter = ax.scatter(
    sample['AQI_lag1'], sample['AQI'],
    c=sample['AQI_label'].astype(int),
    cmap='RdYlGn_r', alpha=0.4, s=10
)
ax.plot([0, 500], [0, 500], 'k--', alpha=0.3, label='y = x')
ax.set_xlabel('前一日 AQI')
ax.set_ylabel('當日 AQI')
ax.set_title('前一日 AQI vs 當日 AQI（隨機 5,000 筆）')
plt.colorbar(scatter, ax=ax, label='AQI 等級')
save_fig('step5_lag1_scatter')

# ── 5.10 特徵相關性熱力圖 ────────────────────────────────────────────────
print("  繪製特徵相關性熱力圖...")
feature_cols = [
    'AQI', 'SI_PM25_max', 'SI_PM10_max', 'SI_O3_max',
    'SI_CO_max', 'SI_SO2_max', 'SI_NO2_max',
    'AMB_TEMP_mean', 'RH_mean', 'WIND_SPEED_mean',
    'RAINFALL_sum', 'AQI_lag1', 'AQI_7d_mean', 'month', 'region_code'
]
feature_cols = [c for c in feature_cols if c in df_model.columns]
corr = df_model[feature_cols].corr()

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr, ax=ax, cmap='coolwarm', center=0,
            annot=True, fmt='.2f', annot_kws={'size': 7},
            linewidths=0.3, square=True)
ax.set_title('特徵相關性矩陣')
save_fig('step5_feature_correlation')

# ── 5.11 週間 vs 週末 AQI 分布 ───────────────────────────────────────────
print("  繪製週間/週末 AQI 比較圖...")
fig, ax = plt.subplots(figsize=(8, 5))
df_model.boxplot(column='AQI', by='is_weekend', ax=ax,
                 patch_artist=True,
                 boxprops=dict(facecolor='#5B9BD5'),
                 medianprops=dict(color='red', linewidth=2))
ax.set_xticklabels(['週間', '週末'])
ax.set_title('週間 vs 週末 AQI 分布')
ax.set_xlabel('')
ax.set_ylabel('AQI')
plt.suptitle('')
save_fig('step5_weekday_weekend_aqi')

# ══════════════════════════════════════════════════════════════════════════════
# 儲存
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'station', 'date', 'year', 'month', 'day_of_week', 'is_weekend', 'season',
    'region', 'region_code',
    'AMB_TEMP_mean', 'RH_mean', 'WIND_SPEED_mean',
    'wind_sin', 'wind_cos', 'RAINFALL_sum',
    'AQI_lag1', 'AQI_lag2', 'AQI_lag3',
    'AQI_label_lag1', 'AQI_7d_mean',
    'AQI', 'AQI_label'
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df_model.columns]
df_final = df_model[FEATURE_COLS].copy()

df_final.to_parquet('aqi_features.parquet', index=False)
print(f"\n儲存完成：aqi_features.parquet")
print(f"檔案大小：{Path('aqi_features.parquet').stat().st_size / 1024**2:.1f} MB")
print(f"\n=== Step 5 完成 ===")
print(f"最終特徵資料集：{len(df_final):,} 筆 × {len(df_final.columns)} 欄")
print(f"\n特徵欄位清單：")
for c in df_final.columns:
    print(f"  {c}")
print(f"\n樣本：")
print(df_final[['station','date','AQI','AQI_label','region',
                'month','AQI_lag1','wind_sin','wind_cos']].head(5).to_string())