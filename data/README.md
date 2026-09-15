# Data

## `raw/telco_churn.csv`
IBM Telco Customer Churn dataset — 7,043 rows, 21 columns.

| Column | Description |
|--------|-------------|
| `customerID` | Unique customer id |
| `gender`, `SeniorCitizen`, `Partner`, `Dependents` | Demographics |
| `tenure` | Months the customer has stayed |
| `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Subscribed services |
| `Contract`, `PaperlessBilling`, `PaymentMethod` | Account information |
| `MonthlyCharges`, `TotalCharges` | Billing amounts |
| `Churn` | **Target** — did the customer leave? (Yes/No) |

## `processed/`
Reserved for any intermediate artifacts produced by the pipeline. Empty by default.
