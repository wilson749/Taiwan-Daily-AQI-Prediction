import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 基本設定
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('./figures/real_test')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def save_fig(name):
    plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 已儲存：figures/real_test/{name}.png")

LABEL_NAMES  = ['Good', 'Moderate', 'Unhealthy(Sen.)',
                'Unhealthy', 'Very Unhealthy', 'Hazardous']
LABEL_COLORS = ['#00E400', '#FFFF00', '#FF7E00',
                '#FF0000', '#8F3F97', '#7E0023']

FEATURE_COLS = [
    'month', 'day_of_week', 'is_weekend', 'season', 'region_code',
    'AMB_TEMP_mean', 'RH_mean', 'WIND_SPEED_mean',
    'wind_sin', 'wind_cos', 'RAINFALL_sum',
    'AQI_lag1', 'AQI_lag2', 'AQI_lag3',
    'AQI_label_lag1', 'AQI_7d_mean',
]
TARGET = 'AQI_label'

# ══════════════════════════════════════════════════════════════════════════════
# 載入資料與重新訓練模型
# ══════════════════════════════════════════════════════════════════════════════
print("載入資料...")
df = pd.read_parquet('aqi_features.parquet')
df = df.sort_values(['station', 'date']).reset_index(drop=True)
df['year']  = df['year'].astype(int)
df[TARGET]  = df[TARGET].astype(int)

train_df = df[df['year'].between(2016, 2023)].copy()
val_df   = df[df['year'] == 2024].copy()
test_df  = df[df['year'] == 2025].copy()

# 填補缺值（以訓練集均值）
for col in FEATURE_COLS:
    col_mean = train_df[col].mean()
    train_df[col] = train_df[col].fillna(col_mean)
    val_df[col]   = val_df[col].fillna(col_mean)
    test_df[col]  = test_df[col].fillna(col_mean)

train_df = train_df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
test_df  = test_df.dropna(subset=FEATURE_COLS + [TARGET]).copy()

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET]

print(f"  訓練集：{len(X_train):,} 筆 | 測試集（2025）：{len(X_test):,} 筆")

# ── 重新訓練三個模型 ──────────────────────────────────────────────────────
print("\n訓練模型...")

lr = LogisticRegression(
    solver='lbfgs', max_iter=1000,
    class_weight='balanced', random_state=42, n_jobs=-1
)
lr.fit(X_train, y_train)
print("  Logistic Regression 完成")

rf = RandomForestClassifier(
    n_estimators=300, max_depth=None, min_samples_split=2,
    class_weight='balanced', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
print("  Random Forest 完成")

class_counts  = y_train.value_counts().to_dict()
sample_weight = y_train.map(
    lambda c: len(y_train) / (6 * class_counts[c])
)
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, objective='multi:softmax', num_class=6,
    eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weight)
print("  XGBoost 完成\n")

# ── 產生預測結果 ──────────────────────────────────────────────────────────
test_df = test_df.copy()
test_df['pred_lr']  = lr.predict(X_test)
test_df['pred_rf']  = rf.predict(X_test)
test_df['pred_xgb'] = xgb_model.predict(X_test)


# ══════════════════════════════════════════════════════════════════════════════
# Real Test 1 — 代表性測站時序預測圖
# ══════════════════════════════════════════════════════════════════════════════
print("Real Test 1：代表性測站時序預測圖...")

# 各地區選一個代表測站
DEMO_STATIONS = {
    '北部（萬里）':  '萬里',
    '中部（忠明）':  '忠明',
    '南部（小港）':  '小港',
    '東部（花蓮）':  '花蓮',
    '離島（金門）':  '金門',
}

fig, axes = plt.subplots(len(DEMO_STATIONS), 1,
                         figsize=(16, 4 * len(DEMO_STATIONS)))

for ax, (label, station) in zip(axes, DEMO_STATIONS.items()):
    stn = test_df[test_df['station'] == station].copy()
    if len(stn) == 0:
        ax.set_title(f'{label} — 無資料')
        continue

    stn = stn.sort_values('date')
    x   = stn['date']

    # 真實值（背景色塊）
    for i, row in stn.iterrows():
        ax.axvspan(
            row['date'] - pd.Timedelta(hours=12),
            row['date'] + pd.Timedelta(hours=12),
            color=LABEL_COLORS[int(row[TARGET])], alpha=0.25
        )

    # 預測線
    ax.plot(x, stn[TARGET],    'k-',  linewidth=1.5,
            label='True Label', zorder=5)
    ax.plot(x, stn['pred_rf'], 'b--', linewidth=1,
            label='RF Prediction', alpha=0.8)
    ax.plot(x, stn['pred_xgb'],'r:',  linewidth=1,
            label='XGB Prediction', alpha=0.8)

    ax.set_title(f'{label}（{station}）— 2025 年 AQI 等級預測 vs 實際',
                 fontsize=11)
    ax.set_yticks(range(6))
    ax.set_yticklabels(LABEL_NAMES, fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('2025 年各地區代表測站 AQI 等級預測（Real Test）',
             fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig('rt1_station_timeseries')


# ══════════════════════════════════════════════════════════════════════════════
# Real Test 2 — 單站單月逐日預測細節表
# ══════════════════════════════════════════════════════════════════════════════
print("Real Test 2：單站單月逐日預測細節...")

# 選小港 2025 年 1 月（南部冬季污染較重）
focus_station = '小港'
focus_month   = 1

detail = test_df[
    (test_df['station'] == focus_station) &
    (test_df['date'].dt.month == focus_month)
].copy().sort_values('date')

detail['True']    = detail[TARGET].map(lambda x: LABEL_NAMES[x])
detail['RF']      = detail['pred_rf'].map(lambda x: LABEL_NAMES[x])
detail['XGB']     = detail['pred_xgb'].map(lambda x: LABEL_NAMES[x])
detail['RF_OK']   = detail[TARGET] == detail['pred_rf']
detail['XGB_OK']  = detail[TARGET] == detail['pred_xgb']

print(f"\n  {focus_station} 2025 年 {focus_month} 月逐日預測結果：")
display_cols = ['date', 'True', 'RF', 'RF_OK', 'XGB', 'XGB_OK']
print(detail[display_cols].to_string(index=False))

# 視覺化
fig, ax = plt.subplots(figsize=(14, 5))
x = np.arange(len(detail))

ax.bar(x, detail[TARGET], color=[LABEL_COLORS[i] for i in detail[TARGET]],
       label='True Label', alpha=0.6, width=0.6)
ax.scatter(x, detail['pred_rf'],  marker='o', s=60, color='blue',
           label='RF Prediction',  zorder=5)
ax.scatter(x, detail['pred_xgb'], marker='^', s=60, color='red',
           label='XGB Prediction', zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(
    [d.strftime('%m/%d') for d in detail['date']],
    rotation=45, fontsize=7
)
ax.set_yticks(range(6))
ax.set_yticklabels(LABEL_NAMES, fontsize=9)
ax.set_title(f'{focus_station} — 2025 年 {focus_month} 月逐日 AQI 等級預測')
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig('rt2_daily_detail')


# ══════════════════════════════════════════════════════════════════════════════
# Real Test 3 — 各測站 2025 年預測準確率地圖（熱力表）
# ══════════════════════════════════════════════════════════════════════════════
print("\nReal Test 3：各測站 2025 年預測準確率...")

station_acc = (
    test_df.groupby('station')
    .apply(lambda g: pd.Series({
        'n':       len(g),
        'acc_rf':  accuracy_score(g[TARGET], g['pred_rf']),
        'acc_xgb': accuracy_score(g[TARGET], g['pred_xgb']),
        'f1_rf':   f1_score(g[TARGET], g['pred_rf'],
                            average='weighted', zero_division=0),
        'f1_xgb':  f1_score(g[TARGET], g['pred_xgb'],
                            average='weighted', zero_division=0),
    }))
    .reset_index()
    .sort_values('acc_rf', ascending=False)
)

print(f"\n  各測站 RF Accuracy（前10 / 後5）：")
print(station_acc[['station','acc_rf','acc_xgb','f1_rf','f1_xgb']]
      .head(10).to_string(index=False))
print("  ...")
print(station_acc[['station','acc_rf','acc_xgb','f1_rf','f1_xgb']]
      .tail(5).to_string(index=False))

# 視覺化：RF vs XGB per-station accuracy
fig, ax = plt.subplots(figsize=(16, 6))
x = np.arange(len(station_acc))
w = 0.35
ax.bar(x - w/2, station_acc['acc_rf'],  w,
       label='Random Forest', color='#2E75B6', alpha=0.85)
ax.bar(x + w/2, station_acc['acc_xgb'], w,
       label='XGBoost',       color='#ED7D31', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(station_acc['station'], rotation=90, fontsize=7)
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1.05)
ax.set_title('各測站 2025 年預測準確率（Real Test）')
ax.axhline(station_acc['acc_rf'].mean(),  color='#2E75B6',
           linestyle='--', alpha=0.6,
           label=f'RF 平均 {station_acc["acc_rf"].mean():.3f}')
ax.axhline(station_acc['acc_xgb'].mean(), color='#ED7D31',
           linestyle='--', alpha=0.6,
           label=f'XGB 平均 {station_acc["acc_xgb"].mean():.3f}')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig('rt3_station_accuracy')


# ══════════════════════════════════════════════════════════════════════════════
# Real Test 4 — 月份別準確率趨勢
# ══════════════════════════════════════════════════════════════════════════════
print("\nReal Test 4：2025 年月份別準確率趨勢...")

monthly_acc = (
    test_df.groupby(test_df['date'].dt.month)
    .apply(lambda g: pd.Series({
        'acc_lr':  accuracy_score(g[TARGET], g['pred_lr']),
        'acc_rf':  accuracy_score(g[TARGET], g['pred_rf']),
        'acc_xgb': accuracy_score(g[TARGET], g['pred_xgb']),
    }))
    .reset_index()
    .rename(columns={'date': 'month'})
)

fig, ax = plt.subplots(figsize=(10, 4))
months = [f'{m}月' for m in monthly_acc['month']]
ax.plot(months, monthly_acc['acc_lr'],  'g-o', label='Logistic Regression')
ax.plot(months, monthly_acc['acc_rf'],  'b-s', label='Random Forest')
ax.plot(months, monthly_acc['acc_xgb'], 'r-^', label='XGBoost')
ax.set_title('2025 年各月份預測準確率趨勢')
ax.set_ylabel('Accuracy')
ax.set_ylim(0.5, 1.05)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
save_fig('rt4_monthly_accuracy')


# ══════════════════════════════════════════════════════════════════════════════
# Real Test 5 — 誤判案例分析
# ══════════════════════════════════════════════════════════════════════════════
print("\nReal Test 5：RF 誤判案例分析...")

errors = test_df[test_df[TARGET] != test_df['pred_rf']].copy()
errors['true_name'] = errors[TARGET].map(lambda x: LABEL_NAMES[x])
errors['pred_name'] = errors['pred_rf'].map(lambda x: LABEL_NAMES[x])

# 誤判方向統計
error_matrix = (
    errors.groupby(['true_name', 'pred_name'])
    .size()
    .reset_index(name='count')
    .sort_values('count', ascending=False)
)

print(f"\n  RF 誤判總數：{len(errors):,} 筆（佔測試集 {len(errors)/len(test_df)*100:.1f}%）")
print(f"\n  主要誤判方向（前 10）：")
print(error_matrix.head(10).to_string(index=False))

# 誤判熱力圖
error_pivot = errors.groupby(['true_name', 'pred_name']).size().unstack(fill_value=0)

fig, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(error_pivot, ax=ax, cmap='Oranges',
            annot=True, fmt='d', linewidths=0.3)
ax.set_title('RF 誤判方向熱力圖（True → Predicted）')
ax.set_xlabel('預測等級')
ax.set_ylabel('真實等級')
plt.tight_layout()
save_fig('rt5_error_heatmap')

# ── 誤判最嚴重的測站 ─────────────────────────────────────────────────────
worst_stations = (
    errors.groupby('station').size()
    .sort_values(ascending=False)
    .head(10)
    .reset_index(name='error_count')
)
worst_stations['total'] = worst_stations['station'].map(
    test_df.groupby('station').size()
)
worst_stations['error_rate'] = (
    worst_stations['error_count'] / worst_stations['total']
)

print(f"\n  誤判率最高的前 10 站：")
print(worst_stations.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(worst_stations['station'], worst_stations['error_rate'],
       color='#C00000', alpha=0.8)
ax.set_title('RF 誤判率最高的前 10 測站（2025 年）')
ax.set_ylabel('誤判率')
ax.set_ylim(0, 1)
ax.tick_params(axis='x', rotation=45)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig('rt5_worst_stations')

# ══════════════════════════════════════════════════════════════════════════════
# Real Test 6 — 地區別效能比較（Fairness Audit）
# ══════════════════════════════════════════════════════════════════════════════
print("\nReal Test 6：地區別效能比較（Fairness Audit）...")

REGION_MAP_AUDIT = {
    '基隆': '北部', '士林': '北部', '中山': '北部', '松山': '北部',
    '萬華': '北部', '古亭': '北部', '大同': '北部', '板橋': '北部',
    '新莊': '北部', '永和': '北部', '土城': '北部', '汐止': '北部',
    '林口': '北部', '淡水': '北部', '萬里': '北部', '富貴角': '北部',
    '陽明': '北部', '新店': '北部', '三重': '北部',
    '桃園': '北部', '中壢': '北部', '平鎮': '北部', '龍潭': '北部',
    '大園': '北部', '觀音': '北部', '湖口': '北部',
    '新竹': '北部', '竹東': '北部', '宜蘭': '北部', '冬山': '北部',
    '頭份': '中部', '苗栗': '中部', '三義': '中部',
    '豐原': '中部', '沙鹿': '中部', '西屯': '中部',
    '忠明': '中部', '大里': '中部',
    '彰化': '中部', '線西': '中部', '二林': '中部',
    '南投': '中部', '埔里': '中部', '竹山': '中部',
    '崙背': '中部', '臺西': '中部', '麥寮': '中部', '斗六': '中部',
    '朴子': '南部', '嘉義': '南部', '新港': '南部',
    '新營': '南部', '善化': '南部', '安南': '南部', '臺南': '南部',
    '左營': '南部', '楠梓': '南部', '橋頭': '南部', '仁武': '南部',
    '大寮': '南部', '林園': '南部', '鳳山': '南部', '前鎮': '南部',
    '小港': '南部', '前金': '南部', '菜寮': '南部', '美濃': '南部',
    '潮州': '南部', '屏東': '南部', '恆春': '南部', '馬公': '南部',
    '花蓮': '東部', '臺東': '東部', '關山': '東部',
    '金門': '離島', '馬祖': '離島',
}

test_df['region_audit'] = test_df['station'].map(REGION_MAP_AUDIT).fillna('其他')
region_order_audit = ['北部', '中部', '南部', '東部', '離島']

region_perf = (
    test_df[test_df['region_audit'].isin(region_order_audit)]
    .groupby('region_audit')
    .apply(lambda g: pd.Series({
        'n_records':   len(g),
        'acc_rf':      accuracy_score(g[TARGET], g['pred_rf']),
        'acc_xgb':     accuracy_score(g[TARGET], g['pred_xgb']),
        'f1_rf':       f1_score(g[TARGET], g['pred_rf'],
                                average='weighted', zero_division=0),
        'f1_xgb':      f1_score(g[TARGET], g['pred_xgb'],
                                average='weighted', zero_division=0),
        # 高風險漏報率：真實 >= 3（Unhealthy 以上）但預測 <= 1（Good/Moderate）
        'miss_rate_rf':  ((g[TARGET] >= 3) & (g['pred_rf']  <= 1)).mean(),
        'miss_rate_xgb': ((g[TARGET] >= 3) & (g['pred_xgb'] <= 1)).mean(),
    }))
    .reindex(region_order_audit)
    .reset_index()
    .rename(columns={'region_audit': 'region'})
)

print("\n  地區別效能摘要：")
print(region_perf.round(4).to_string(index=False))

# ── 視覺化 1：地區別 Accuracy & F1 ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(region_perf))
w = 0.35

for ax, (rf_col, xgb_col, title) in zip(axes, [
    ('acc_rf',  'acc_xgb', 'Accuracy by Region'),
    ('f1_rf',   'f1_xgb',  'Weighted F1 by Region'),
]):
    bars_rf  = ax.bar(x - w/2, region_perf[rf_col],  w,
                      label='Random Forest', color='#2E75B6', alpha=0.85)
    bars_xgb = ax.bar(x + w/2, region_perf[xgb_col], w,
                      label='XGBoost',       color='#ED7D31', alpha=0.85)
    for bar, v in zip(list(bars_rf) + list(bars_xgb),
                      list(region_perf[rf_col]) + list(region_perf[xgb_col])):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{v:.3f}', ha='center', fontsize=8)
    ymin = min(region_perf[rf_col].min(), region_perf[xgb_col].min()) - 0.05
    ax.set_ylim(max(0, ymin), 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(region_perf['region'])
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('地區別模型效能比較（Fairness Audit）', fontsize=13)
plt.tight_layout()
save_fig('rt6_region_performance')

# ── 視覺化 2：高風險漏報率（Miss Rate）────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
bars_rf  = ax.bar(x - w/2, region_perf['miss_rate_rf'],  w,
                  label='Random Forest', color='#2E75B6', alpha=0.85)
bars_xgb = ax.bar(x + w/2, region_perf['miss_rate_xgb'], w,
                  label='XGBoost',       color='#ED7D31', alpha=0.85)
for bar, v in zip(list(bars_rf) + list(bars_xgb),
                  list(region_perf['miss_rate_rf']) +
                  list(region_perf['miss_rate_xgb'])):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.001,
            f'{v:.3f}', ha='center', fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(region_perf['region'])
ax.set_ylabel(r'Miss Rate (True $\geq$ Unhealthy, Pred $\leq$ Moderate)')
ax.set_title(r'高風險漏報率（AQI $\geq$ Unhealthy 被預測為 Good/Moderate）')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
save_fig('rt6_region_miss_rate')


print(f"\n{'='*60}")
print("Real Test 全部完成，圖表已儲存至 ./figures/real_test/")
print('='*60)