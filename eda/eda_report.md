# Exploratory Data Analysis: 04_smart_building_energy_digital_twin2.csv

## Dataset overview
- Rows: 500
- Columns: 14
- Date range: 2026-01-01 to 2027-05-15
- Unique Building_ID values: 500
- Exact duplicate rows: 0

## Data quality
### Column types
|                             | dtype          |
|:----------------------------|:---------------|
| Building_ID                 | object         |
| Date                        | datetime64[ns] |
| Hour                        | int64          |
| Temperature                 | float64        |
| Humidity                    | float64        |
| Occupancy                   | int64          |
| Lighting_Load               | float64        |
| HVAC_Load                   | float64        |
| Computer_Load               | float64        |
| Other_Load                  | float64        |
| Previous_Energy_Consumption | float64        |
| Weekend                     | int64          |
| Holiday                     | int64          |
| Total_Energy_Consumption    | float64        |

### Missing values
|                             |   missing |
|:----------------------------|----------:|
| Building_ID                 |         0 |
| Date                        |         0 |
| Hour                        |         0 |
| Temperature                 |         0 |
| Humidity                    |         0 |
| Occupancy                   |         0 |
| Lighting_Load               |         0 |
| HVAC_Load                   |         0 |
| Computer_Load               |         0 |
| Other_Load                  |         0 |
| Previous_Energy_Consumption |         0 |
| Weekend                     |         0 |
| Holiday                     |         0 |
| Total_Energy_Consumption    |         0 |

### IQR outlier counts (exploratory flags; rows are retained)
|                             |   outlier_count |
|:----------------------------|----------------:|
| Hour                        |               0 |
| Temperature                 |               0 |
| Humidity                    |               6 |
| Occupancy                   |               0 |
| Lighting_Load               |               2 |
| HVAC_Load                   |               2 |
| Computer_Load               |               0 |
| Other_Load                  |               1 |
| Previous_Energy_Consumption |               3 |
| Weekend                     |               0 |
| Holiday                     |              29 |
| Total_Energy_Consumption    |               1 |

## Numeric summary
|                             |   count | mean                | min                 | 25%                 | 50%                 | 75%                 | max                 |        std |
|:----------------------------|--------:|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|:--------------------|-----------:|
| Date                        |     500 | 2026-09-07 12:00:00 | 2026-01-01 00:00:00 | 2026-05-05 18:00:00 | 2026-09-07 12:00:00 | 2027-01-10 06:00:00 | 2027-05-15 00:00:00 | nan        |
| Hour                        |     500 | 11.338              | 0.0                 | 5.0                 | 11.0                | 18.0                | 23.0                |   7.20191  |
| Temperature                 |     500 | 29.0494             | 18.0                | 25.7                | 29.15               | 32.425              | 42.0                |   4.84949  |
| Humidity                    |     500 | 62.5784             | 25.0                | 54.075              | 62.95               | 71.525              | 95.0                |  13.3794   |
| Occupancy                   |     500 | 55.994              | 0.0                 | 40.0                | 56.0                | 72.0                | 100.0               |  23.576    |
| Lighting_Load               |     500 | 56.833400000000005  | 0.0                 | 42.575              | 57.55               | 70.92500000000001   | 100.0               |  20.5893   |
| HVAC_Load                   |     500 | 46.237199999999994  | 0.0                 | 34.6                | 46.9                | 58.025              | 96.4                |  18.2336   |
| Computer_Load               |     500 | 38.785              | 0.0                 | 28.375              | 39.3                | 48.0                | 77.4                |  14.4423   |
| Other_Load                  |     500 | 19.9336             | 2.0                 | 14.475              | 19.9                | 25.3                | 42.6                |   7.95971  |
| Previous_Energy_Consumption |     500 | 619.304             | 80.0                | 490.575             | 611.0               | 748.05              | 1134.6              | 180.997    |
| Weekend                     |     500 | 0.286               | 0.0                 | 0.0                 | 0.0                 | 1.0                 | 1.0                 |   0.452342 |
| Holiday                     |     500 | 0.058               | 0.0                 | 0.0                 | 0.0                 | 0.0                 | 1.0                 |   0.233978 |
| Total_Energy_Consumption    |     500 | 1161.0198000000003  | 218.1               | 910.775             | 1160.7              | 1397.1499999999999  | 2137.9              | 349.244    |

## Interpretation
- Building_ID is unique for every row and is excluded from model features to avoid memorizing identifiers.
- Date is transformed into month, day-of-week, day-of-year, and cyclic hour/month features.
- The small IQR-flagged values are retained because clipping would alter valid energy/load observations.
- Dataset rows are chronological observations; model validation uses the later 20% as a holdout.

## Visualizations
- [correlation_heatmap](./correlation_heatmap.png)
- [energy_distribution](./energy_distribution.png)
- [individual_load_analysis](./individual_load_analysis.png)
- [occupancy_vs_energy](./occupancy_vs_energy.png)
- [temperature_vs_energy](./temperature_vs_energy.png)
- [time_based_analysis](./time_based_analysis.png)
