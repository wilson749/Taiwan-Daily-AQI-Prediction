# Predicting AQI Risk Levels for Sustainable Cities (SDG 11)

A machine learning pipeline for predicting daily Air Quality Index (AQI)
risk levels across 77 monitoring stations in Taiwan (2016–2025), aligned
with United Nations Sustainable Development Goal 11 (Sustainable Cities
and Communities).

---

## Project Overview

This project builds a multi-class classification system that predicts
Taiwan's daily AQI risk level (6 categories: Good → Hazardous) using
meteorological observations and lag-based AQI history. Three models are
compared: Logistic Regression (baseline), Random Forest (primary), and
XGBoost (advanced).

**Key results on 2025 held-out test set:**

| Model               | Accuracy | Weighted F1 | Hazardous Recall |
|---------------------|----------|-------------|-----------------|
| Logistic Regression | 0.629    | 0.657       | 0.39            |
| Random Forest       | 0.806    | 0.793       | 0.22            |
| XGBoost             | 0.749    | 0.767       | **0.79**        |

XGBoost is recommended for public health alerting systems due to its
substantially higher recall on hazardous AQI events.

---

## Data Source

Raw data is sourced from the **Taiwan Ministry of Environment (MOENV)
Open Data Platform** and is included in this repository.

- **Coverage**: 77 monitoring stations, 2016–2025
- **Format**: CSV (2018–2025) and XLS (2016–2017)
- **Download**: https://data.moenv.gov.tw/

After downloading, organize files under `./data/` in the following
structure:
```
data/
├── 全部_2018/
│   ├── 三義_2018.csv
│   └── ...
├── 全部_2019/
│   └── ...
├── 103_HOUR_00_20170317/
│   └── 103年 高屏空品區/
│       └── 103年小港站_20170317.xls
└── ...
```
---

## Project Structure
```
.
├── data/                        # Raw data (not tracked by git)
├── figures/                     # Generated plots (not tracked by git)
│   └── real_test/
├── coverage_audit.csv           # Station-year coverage audit results
├── usable_station_years.csv     # Valid station-year combinations
│
├── 1_NR_Combinations.py         # Analyse NR marker patterns
├── 2_Stations_Audit.py          # Coverage audit across stations/years
├── 3_Reshape_Data.py            # Load, reshape, clean → aqi_hourly_tidy.parquet
├── 4_Cleaning_Fill_Data.py      # Forward-fill + AQI label generation → aqi_daily.parquet
├── 5_Feature_Engineering.py     # Feature engineering → aqi_features.parquet
├── 6_Model_Training.py          # Train LR / RF / XGBoost, evaluation
├── 7_Real_Test.py               # 2025 held-out test, auditing analysis
│
├── requirements.txt
├── .gitignore
└── README.md
```
---

## Pipeline Execution Order

Install dependencies first:

```bash
pip install -r requirements.txt
```

Then run scripts in order:

```bash
python 1_NR_Combinations.py      # Optional: analyse NR marker patterns
python 2_Stations_Audit.py       # Generate coverage_audit.csv and usable_station_years.csv
python 3_Reshape_Data.py         # Outputs: aqi_hourly_tidy.parquet
python 4_Cleaning_Fill_Data.py   # Outputs: aqi_daily.parquet + figures/step3_*.png, step4_*.png
python 5_Feature_Engineering.py  # Outputs: aqi_features.parquet + figures/step5_*.png
python 6_Model_Training.py       # Outputs: figures/*.png (confusion matrices, feature importance)
python 7_Real_Test.py            # Outputs: figures/real_test/*.png
```

> **Note**: Steps 1–2 only need to run once to generate the station audit
> files. Steps 3–5 can be skipped on subsequent runs if the `.parquet`
> files already exist.

---

## AQI Risk Level Classification

| Label | AQI Range | Risk Level                    |
|-------|-----------|-------------------------------|
| 0     | 0–50      | Good                          |
| 1     | 51–100    | Moderate                      |
| 2     | 101–150   | Unhealthy for Sensitive Groups|
| 3     | 151–200   | Unhealthy                     |
| 4     | 201–300   | Very Unhealthy                |
| 5     | 301+      | Hazardous                     |

AQI labels are computed programmatically from raw pollutant measurements
using Taiwan MOENV's official piecewise linear interpolation breakpoints,
following the Daily AQI (日 AQI) standard.

---

## Feature Set

| Feature           | Type          | Description                                   |
|-------------------|---------------|-----------------------------------------------|
| month             | Temporal      | Month of year (1–12)                          |
| day_of_week       | Temporal      | Day of week (0=Mon, 6=Sun)                    |
| is_weekend        | Temporal      | Weekend binary flag                           |
| season            | Temporal      | Winter(0)/Spring(1)/Summer(2)/Autumn(3)       |
| region_code       | Spatial       | North(0)/Central(1)/South(2)/East(3)/Island(4)|
| AMB_TEMP_mean     | Meteorological| Daily mean temperature (°C)                   |
| RH_mean           | Meteorological| Daily mean relative humidity (%)              |
| WIND_SPEED_mean   | Meteorological| Daily mean wind speed (m/s)                   |
| wind_sin/cos      | Directional   | Wind direction circular encoding              |
| RAINFALL_sum      | Meteorological| Daily cumulative rainfall (mm)                |
| AQI_lag1/2/3      | Lag           | Previous 1–3 day AQI values                   |
| AQI_label_lag1    | Lag           | Previous day AQI risk level                   |
| AQI_7d_mean       | Rolling       | 7-day rolling mean AQI                        |

---

## SDG 11 Alignment

This project directly addresses **SDG 11 Target 11.6**: reducing the
adverse environmental impact of cities, particularly regarding air
quality. A reliable daily AQI prediction system enables:

- City governments to issue early warnings one day in advance
- Health authorities to activate emergency response protocols
- Vulnerable populations (elderly, children, respiratory patients)
  to take timely protective measures

The geographic fairness audit reveals that northern Taiwan (population
~7 million) faces the highest high-risk miss rate under Random Forest
(4.84%), highlighting the need for model selection tailored to regional
public health risk profiles.

---

## Course Information

- **Course**: Artificial intelligence development and security
- **Institution**: National Taipei University of Technology
- **Instructor**: Professor 彭祖乙
- **Submission**: Personal Midterm Report(May 18, 2026)