import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 基本設定
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

OUTPUT_DIR = Path('./figures')
OUTPUT_DIR.mkdir(exist_ok=True)

def save_fig(name):
    plt.savefig(OUTPUT_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → 已儲存：figures/{name}.png")

LABEL_NAMES  = ['Good', 'Moderate', 'Unhealthy(Sen.)',
                'Unhealthy', 'Very Unhealthy', 'Hazardous']
LABEL_COLORS = ['#00E400','#FFFF00','#FF7E00','#FF0000','#8F3F97','#7E0023']

# ══════════════════════════════════════════════════════════════════════════════
# 載入資料
# ══════════════════════════════════════════════════════════════════════════════
print("載入 aqi_features.parquet...")
df = pd.read_parquet('aqi_features.parquet')
df = df.sort_values(['station', 'date']).reset_index(drop=True)
df['year'] = df['year'].astype(int)
print(f"載入完成：{len(df):,} 筆 × {len(df.columns)} 欄\n")

# ══════════════════════════════════════════════════════════════════════════════
# Feature / Target 定義
# ══════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'month', 'day_of_week', 'is_weekend', 'season', 'region_code',
    'AMB_TEMP_mean', 'RH_mean', 'WIND_SPEED_mean',
    'wind_sin', 'wind_cos', 'RAINFALL_sum',
    'AQI_lag1', 'AQI_lag2', 'AQI_lag3',
    'AQI_label_lag1', 'AQI_7d_mean',
]
TARGET = 'AQI_label'

df[TARGET] = df[TARGET].astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Train / Validation / Test Split
# ══════════════════════════════════════════════════════════════════════════════
print("Step 6：資料切割...")
train_df = df[df['year'].between(2016, 2023)].copy()
val_df   = df[df['year'] == 2024].copy()
test_df  = df[df['year'] == 2025].copy()

# ── 其他欄位缺值：以訓練集均值填補 ──────────────────────────────────────
for col in FEATURE_COLS:
    if col == 'SI_O3_max':
        continue
    col_mean = train_df[col].mean()
    train_df[col] = train_df[col].fillna(col_mean)
    val_df[col]   = val_df[col].fillna(col_mean)
    test_df[col]  = test_df[col].fillna(col_mean)

# ── dropna 與切割 ────────────────────────────────────────────────────────
train_df = train_df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
val_df   = val_df.dropna(subset=FEATURE_COLS + [TARGET]).copy()
test_df  = test_df.dropna(subset=FEATURE_COLS + [TARGET]).copy()

X_train, y_train = train_df[FEATURE_COLS], train_df[TARGET]
X_val,   y_val   = val_df[FEATURE_COLS],   val_df[TARGET]
X_test,  y_test  = test_df[FEATURE_COLS],  test_df[TARGET]

print(f"\n  訓練集：{len(X_train):,} 筆（2016–2023）")
print(f"  驗證集：{len(X_val):,} 筆（2024）")
print(f"  測試集：{len(X_test):,} 筆（2025）")
print(f"\n  訓練集 AQI 等級分布：")
print(y_train.value_counts().sort_index().to_dict())

# ── 視覺化：三分集類別分布 ────────────────────────────────────────────────
print("\n  繪製資料集分布圖...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
for ax, (subset_y, name) in zip(axes, [
    (y_train, '訓練集 2016–2023'),
    (y_val,   '驗證集 2024'),
    (y_test,  '測試集 2025'),
]):
    counts = subset_y.value_counts().sort_index()
    colors = [LABEL_COLORS[i] for i in counts.index]
    ax.bar([LABEL_NAMES[i] for i in counts.index], counts.values, color=colors)
    ax.set_title(name)
    ax.set_ylabel('筆數')
    ax.tick_params(axis='x', rotation=35)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 30, f'{v:,}', ha='center', fontsize=7)
plt.suptitle('訓練 / 驗證 / 測試集 AQI 等級分布', fontsize=13)
plt.tight_layout()
save_fig('step6_split_distribution')


# ══════════════════════════════════════════════════════════════════════════════
# 共用函式
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    wf1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\n{'='*55}")
    print(f"[{name}]")
    print(f"Accuracy: {acc:.4f}   |   Weighted F1: {wf1:.4f}")
    print('='*55)
    print(classification_report(
        y_true, y_pred,
        labels=list(range(6)),
        target_names=LABEL_NAMES,
        zero_division=0
    ))
    return acc, wf1

def plot_cm(name, y_true, y_pred, filename):
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ['d', '.1f'],
        [f'{name} — Confusion Matrix（筆數）',
         f'{name} — Confusion Matrix（列百分比 %）']
    ):
        sns.heatmap(data, ax=ax, annot=True, fmt=fmt,
                    cmap='Blues', linewidths=0.3,
                    xticklabels=LABEL_NAMES,
                    yticklabels=LABEL_NAMES)
        ax.set_xlabel('預測值')
        ax.set_ylabel('真實值')
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=35)
        ax.tick_params(axis='y', rotation=0)
    plt.tight_layout()
    save_fig(filename)

def plot_feature_importance(name, importances, filename, color):
    fi = pd.Series(importances, index=FEATURE_COLS).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    fi.plot(kind='bar', ax=ax, color=color, edgecolor='white')
    ax.set_title(f'{name} — Feature Importance')
    ax.set_ylabel('Importance')
    ax.axvline(x=len(fi)//2, color='grey', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_fig(filename)
    print(f"\n  {name} Top 10 Features:")
    print(fi.head(10).to_string())

def plot_highrisk_recall(models_dict, filename):
    """
    models_dict = {'Model Name': (model, X_test, y_test), ...}
    高風險等級（2–5）的 Recall 比較圖
    """
    high_risk_idx   = [2, 3, 4, 5]
    high_risk_names = [LABEL_NAMES[i] for i in high_risk_idx]
    model_colors    = ['#5B9BD5', '#70AD47', '#ED7D31']

    recall_rows = []
    for (model_name, (model, X, y)), color in zip(
        models_dict.items(), model_colors
    ):
        report = classification_report(
            y, model.predict(X),
            labels=list(range(6)),
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0
        )
        for idx in high_risk_idx:
            recall_rows.append({
                'Model':  model_name,
                'Class':  LABEL_NAMES[idx],
                'Recall': report[LABEL_NAMES[idx]]['recall'],
                'Color':  color,
            })

    recall_df = pd.DataFrame(recall_rows)
    x  = np.arange(len(high_risk_names))
    w  = 0.25
    model_names = list(models_dict.keys())

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (model_name, color) in enumerate(
        zip(model_names, model_colors)
    ):
        vals = recall_df[recall_df['Model'] == model_name]['Recall'].values
        bars = ax.bar(x + i*w, vals, w, label=model_name,
                      color=color, edgecolor='white')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f'{v:.2f}', ha='center', fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(high_risk_names, rotation=15)
    ax.set_ylabel('Recall')
    ax.set_ylim(0, 1.15)
    ax.set_title('高風險 AQI 等級 Recall 比較（Cost-sensitive Evaluation）')
    ax.axhline(0.5, color='red', linestyle='--',
               alpha=0.4, label='Recall = 0.5 基準線')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    save_fig(filename)


# ══════════════════════════════════════════════════════════════════════════════
# Step 7a — Logistic Regression（Baseline）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Step 7a：Logistic Regression（Baseline）")
print("="*60)

lr = LogisticRegression(
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
lr.fit(X_train, y_train)

print("\n  --- 驗證集 ---")
lr_val_acc,  lr_val_f1  = evaluate('LR — Val',  y_val,  lr.predict(X_val))
print("\n  --- 測試集 ---")
lr_test_acc, lr_test_f1 = evaluate('LR — Test', y_test, lr.predict(X_test))
plot_cm('Logistic Regression', y_test, lr.predict(X_test), 'lr_cm')


# ══════════════════════════════════════════════════════════════════════════════
# Step 7b — Random Forest
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Step 7b：Random Forest")
print("="*60)
print("""
  Note: Due to computational constraints and project scope,
  expanding-window cross-validation was not implemented.
  A strict chronological train-validation-test split was
  adopted to preserve temporal consistency and avoid data leakage.
""")

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

print("\n  --- 驗證集 ---")
rf_val_acc,  rf_val_f1  = evaluate('RF — Val',  y_val,  rf.predict(X_val))
print("\n  --- 測試集 ---")
rf_test_acc, rf_test_f1 = evaluate('RF — Test', y_test, rf.predict(X_test))
plot_cm('Random Forest', y_test, rf.predict(X_test), 'rf_cm')
plot_feature_importance('Random Forest', rf.feature_importances_,
                        'rf_feature_importance', '#2E75B6')


# ══════════════════════════════════════════════════════════════════════════════
# Step 7c — XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Step 7c：XGBoost")
print("="*60)

class_counts  = y_train.value_counts().to_dict()
sample_weight = y_train.map(
    lambda c: len(y_train) / (6 * class_counts[c])
)

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    objective='multi:softmax',
    num_class=6,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
xgb_model.fit(X_train, y_train, sample_weight=sample_weight)

print("\n  --- 驗證集 ---")
xgb_val_acc,  xgb_val_f1  = evaluate('XGB — Val',  y_val,  xgb_model.predict(X_val))
print("\n  --- 測試集 ---")
xgb_test_acc, xgb_test_f1 = evaluate('XGB — Test', y_test, xgb_model.predict(X_test))
plot_cm('XGBoost', y_test, xgb_model.predict(X_test), 'xgb_cm')
plot_feature_importance('XGBoost', xgb_model.feature_importances_,
                        'xgb_feature_importance', '#ED7D31')


# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — 三模型比較
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Step 8：三模型比較")
print("="*60)

results = pd.DataFrame({
    'Model':    ['Logistic Regression', 'Random Forest', 'XGBoost'],
    'Val Acc':  [lr_val_acc,   rf_val_acc,   xgb_val_acc],
    'Val F1':   [lr_val_f1,    rf_val_f1,    xgb_val_f1],
    'Test Acc': [lr_test_acc,  rf_test_acc,  xgb_test_acc],
    'Test F1':  [lr_test_f1,   rf_test_f1,   xgb_test_f1],
}).round(4)

print("\nMODEL COMPARISON SUMMARY")
print(results.to_string(index=False))

# ── 比較長條圖（動態 y 軸）────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
model_colors = ['#5B9BD5', '#70AD47', '#ED7D31']
x = np.arange(3)
w = 0.35

for ax, (val_col, test_col, title) in zip(axes, [
    ('Val F1',  'Test F1',  'Weighted F1 Score Comparison'),
    ('Val Acc', 'Test Acc', 'Accuracy Comparison'),
]):
    bars_val  = ax.bar(x - w/2, results[val_col],  w,
                       label=val_col,  color='#5B9BD5', edgecolor='white')
    bars_test = ax.bar(x + w/2, results[test_col], w,
                       label=test_col, color='#E07B6A', edgecolor='white')
    for bar, v in zip(list(bars_val) + list(bars_test),
                      list(results[val_col]) + list(results[test_col])):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.002,
                f'{v:.4f}', ha='center', fontsize=7, fontweight='bold')
    ymin = min(results[val_col].min(), results[test_col].min()) - 0.02
    ax.set_ylim(max(0, ymin), 1.01)
    ax.set_xticks(x)
    ax.set_xticklabels(results['Model'], rotation=15)
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('三模型效能比較（驗證集 vs 測試集）', fontsize=13)
plt.tight_layout()
save_fig('model_comparison')

# ── High-risk Recall 比較圖 ───────────────────────────────────────────────
print("\n  繪製 High-risk Recall 比較圖...")
plot_highrisk_recall(
    {
        'Logistic Regression': (lr,        X_test, y_test),
        'Random Forest':       (rf,        X_test, y_test),
        'XGBoost':             (xgb_model, X_test, y_test),
    },
    'highrisk_recall'
)

print(f"\n{'='*60}")
print("全部完成，圖表已儲存至 ./figures/")
print('='*60)