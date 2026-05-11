# Snowflake, Snowpark et Machine Learning

## Objectif

Créer un mini pipeline MLOps avec Snowflake :

```text
Données marketing
- Feature engineering avec Snowpark
- Entraînement d’un modèle ML
- Prédiction de revenus
- Sauvegarde des résultats dans Snowflake
```

---

# 1 — Préparer l’environnement Python

## Anaconda Prompt

```bash
conda activate snowpark310
```

Installer les dépendances :

```bash
pip install pandas pyarrow scikit-learn joblib snowflake-snowpark-python
```

Vérification :

```bash
python -c "import pandas, pyarrow, sklearn, joblib; print('OK')"
```

Lancer Jupyter :

```bash
jupyter notebook
```

---

# 2 — Connexion à Snowflake

## Jupyter Notebook

```python
from snowflake.snowpark import Session
```

```python
connection_parameters = {
    "account": "TON_COMPTE",
    "user": "TON_USER",
    "password": "TON_PASSWORD",
    "role": "ACCOUNTADMIN",
    "warehouse": "COMPUTE_WH"
}

session = Session.builder.configs(connection_parameters).create()
```

Vérification :

```python
session.sql("""
SELECT 
    CURRENT_ROLE(), 
    CURRENT_DATABASE(), 
    CURRENT_SCHEMA(), 
    CURRENT_WAREHOUSE()
""").show()
```

---

# 3 — Créer la base et le schéma

```python
session.sql("CREATE DATABASE IF NOT EXISTS SNOWPARK").collect()
session.sql("CREATE SCHEMA IF NOT EXISTS SNOWPARK.SAMPLE_DATA").collect()

session.sql("USE DATABASE SNOWPARK").collect()
session.sql("USE SCHEMA SAMPLE_DATA").collect()
session.sql("USE WAREHOUSE COMPUTE_WH").collect()
```

---

# 4 — Créer la table CAMPAIGN_SPEND

```python
session.sql("""
CREATE OR REPLACE TABLE CAMPAIGN_SPEND (
    DATE DATE,
    CHANNEL STRING,
    TOTAL_COST FLOAT,
    CLICKS INTEGER,
    ADS_SERVED INTEGER
)
""").collect()
```

Insertion des données :

```python
session.sql("""
INSERT INTO CAMPAIGN_SPEND VALUES
('2024-01-01', 'search_engine', 1000, 120, 10000),
('2024-01-01', 'email', 500, 80, 7000),
('2024-01-01', 'video', 700, 60, 9000),
('2024-01-01', 'social_media', 300, 40, 5000)
""").collect()
```

Afficher les données :

```python
snow_df_spend = session.table("CAMPAIGN_SPEND")
snow_df_spend.show()
```

---

# 5 — Créer la table MONTHLY_REVENUE

```python
session.sql("""
CREATE OR REPLACE TABLE MONTHLY_REVENUE (
    YEAR INTEGER,
    MONTH INTEGER,
    REVENUE FLOAT
)
""").collect()
```

Insertion :

```python
session.sql("""
INSERT INTO MONTHLY_REVENUE VALUES
(2024, 1, 10000),
(2024, 2, 12500),
(2024, 3, 14500)
""").collect()
```

Afficher :

```python
session.table("MONTHLY_REVENUE").show()
```

---

# 6 — Feature Engineering avec Snowpark

```python
from snowflake.snowpark.functions import col, sum as sum_, year, month
```

```python
snow_df_spend = session.table("CAMPAIGN_SPEND")

spend_by_channel = (
    snow_df_spend
    .group_by(
        year(col("DATE")).alias("YEAR"),
        month(col("DATE")).alias("MONTH"),
        col("CHANNEL")
    )
    .agg(sum_(col("TOTAL_COST")).alias("TOTAL_COST"))
)
```

---

# 7 — Jointure avec les revenus

```python
snow_df_revenue = session.table("MONTHLY_REVENUE")

training_df = spend_by_channel.join(
    snow_df_revenue,
    on=["YEAR", "MONTH"]
)
```

---

# 8 — Créer la table MARKETING_BUDGETS_FEATURES

```python
features_df = training_df.dropna()

features_df.write.mode("overwrite").save_as_table(
    "MARKETING_BUDGETS_FEATURES"
)
```

---

# 9 — Charger les données en pandas

```python
df = session.table(
    "SNOWPARK.SAMPLE_DATA.MARKETING_BUDGETS_FEATURES"
).to_pandas()
```

---

# 10 — Préparer les features et la target

```python
X = df.drop("REVENUE", axis=1)
y = df["REVENUE"]
```

---

# 11 — Séparer train/test

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

---

# 12 — Modèle : Régression Linéaire

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

---

# 13 — Évaluer le modèle

```python
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("RMSE :", rmse)
print("R2 :", r2)
```

---

# 14 — Sauvegarder le modèle

```python
import joblib

joblib.dump(model, "revenue_model.joblib")
```

---

# 15 — Faire des prédictions

```python
new_campaigns = [
    [2200, 1100, 1400, 900]
]
```

```python
import pandas as pd

columns = [
    "SEARCH_ENGINE",
    "EMAIL",
    "VIDEO",
    "SOCIAL_MEDIA"
]

new_campaigns_df = pd.DataFrame(
    new_campaigns,
    columns=columns
)
```

```python
predictions = model.predict(new_campaigns_df)

new_campaigns_df["PREDICTED_REVENUE"] = predictions
```

---

# 16 — Sauvegarder les prédictions dans Snowflake

```python
predictions_snow_df = session.create_dataframe(new_campaigns_df)

predictions_snow_df.write.mode("overwrite").save_as_table(
    "CAMPAIGN_REVENUE_PREDICTIONS"
)
```

---

# 17 — Voir les données dans Snowflake

```sql
SHOW TABLES;
```

```sql
SELECT *
FROM CAMPAIGN_SPEND;
```

```sql
SELECT *
FROM CAMPAIGN_REVENUE_PREDICTIONS;
```

---
