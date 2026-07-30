# Task

Predict the final awarded price (`Final_Price_EUR`) for each public procurement tender identified by `Procurement_ID`.

# Metric

Root Mean Squared Error (RMSE) on `Final_Price_EUR`. Lower is better.

```
RMSE = sqrt( mean( (y_true - y_pred)^2 ) )
```

# Submission Format

For every `Procurement_ID` in the test extracts, predict `Final_Price_EUR`. The file must contain a header and follow:

```
Procurement_ID,Final_Price_EUR
PROC-0001,1250000.00
PROC-0002,880450.50
PROC-0003,2100000.00
```

# Dataset

Features are provided as two extracts from a procurement information system. Record order is not aligned across extracts; use `Procurement_ID` to combine them.

- **tenders_profile.csv** — training procedural / buyer / classification attributes, plus `Final_Price_EUR`
- **tenders_commercial.csv** — training commercial / schedule / estimate attributes for the same tenders
- **test_profile.csv** — test procedural / buyer / classification attributes (no target)
- **test_commercial.csv** — test commercial / schedule / estimate attributes (no target)
- **sample_submission.csv** — example submission in the required format

## Columns (shared ideas)

- `Procurement_ID` — unique tender identifier present in both extracts
- `Final_Price_EUR` — final awarded price in euros (training profile extract only; this is the prediction target)
- Remaining columns are tender attributes from each extract (numeric and categorical)
