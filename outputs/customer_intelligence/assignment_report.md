# Customer Intelligence System - Country Segmentation

## Objective
Build an end-to-end intelligence pipeline using clustering and classification to identify priority country/customer segments from socio-economic indicators.

## Dataset
- Records: 167 countries
- Features: 9 numeric indicators
- Missing values: 0

## Best Classification Model
Logistic Regression with test accuracy 1.000

## Segment Profiles
```
                    child_mort  exports  health  imports    income  inflation  life_expec  total_fer      gdpp  priority_rank
segment                                                                                                                      
High Priority            92.96    29.15    6.39    42.32   3942.40      12.02       59.19       5.01   1922.38            1.0
Developing               21.93    40.24    6.20    47.47  12305.60       7.60       72.81       2.31   6486.45            2.0
Stable / Developed        5.00    58.74    8.81    51.49  45672.22       2.67       80.13       1.75  42494.44            3.0
```

## Top High Priority Countries
```
                 country  child_mort  income  life_expec  total_fer  gdpp
                   Haiti       208.0    1500        32.1       3.33   662
            Sierra Leone       160.0    1220        55.0       5.20   399
                    Chad       150.0    1930        56.5       6.59   897
Central African Republic       149.0     888        47.5       5.21   446
                    Mali       137.0    1870        59.5       6.55   708
                 Nigeria       130.0    5150        60.5       5.84  2330
                   Niger       123.0     814        58.8       7.49   348
                  Angola       119.0    5900        60.1       6.16  3530
        Congo, Dem. Rep.       116.0     609        57.5       6.54   334
            Burkina Faso       116.0    1430        57.9       5.87   575
           Guinea-Bissau       114.0    1390        55.6       5.05   547
                   Benin       111.0    1820        61.8       5.36   758
           Cote d'Ivoire       111.0    2690        56.3       5.27  1220
       Equatorial Guinea       111.0   33700        60.9       5.21 17100
                  Guinea       109.0    1190        58.0       5.34   648
```

## Output Files
- country_segments.csv
- segment_profiles.csv
- classification_model_comparison.csv
- classification_reports.txt
- best_classifier.joblib
- EDA, clustering, PCA, DBSCAN, and confusion matrix charts
