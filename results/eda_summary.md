# EDA summary
Rows: 149043, Columns: 14
## Dtypes
Time             int64
Protcol            str
Flag               str
Family             str
Clusters         int64
SeddAddress        str
ExpAddress         str
BTC              int64
USD              int64
Netflow_Bytes    int64
IPaddress          str
Threats            str
Port             int64
Prediction         str

## Missing values
Total missing cells: 0

## Duplicate rows: 0

## Negative `Time` values: 139
## `ExpAddress` == '1' rows: 73
## `Threats` == 'Bonet': 16523, == 'NerisBonet': 6241 (likely misspellings of Botnet/NerisBotnet)

## Cardinality per column
Protcol             3
Prediction          3
IPaddress           4
Port                4
SeddAddress         6
ExpAddress          7
Flag                9
Clusters            9
Threats             9
Family             17
Time               87
BTC              1087
USD              5267
Netflow_Bytes    5818

## Target (`Prediction`) distribution
            count    pct
Prediction              
S           66380  44.54
A           42561  28.56
SS          40102  26.91

## Mutual information vs `Prediction` (higher = more predictive)
USD              0.751401
BTC              0.584629
Netflow_Bytes    0.499406
Flag             0.461745
ExpAddress       0.250738
Clusters         0.223910
SeddAddress      0.189159
Threats          0.183029
Port             0.096925
Family           0.055885
IPaddress        0.033280
Time             0.022371
Protcol          0.011714

## Highly correlated feature pairs (|corr| > 0.85, excluding target)
(none above threshold)
