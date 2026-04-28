import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. загрузка мастер-таблицы
# ---------------------------------------------------------

df = pd.read_csv("master_table_russia_2014_01_rich_model.csv", parse_dates=["date"])
df = df.sort_values("date").set_index("date")

# ---------------------------------------------------------
# 2. лаги инфляции
# ---------------------------------------------------------

df["cpi_lag1"] = df["cpi_mm"].shift(1)
df["cpi_lag2"] = df["cpi_mm"].shift(2)
df["cpi_lag3"] = df["cpi_mm"].shift(3)

# ---------------------------------------------------------
# 3. лог-разности (если уровни)
# ---------------------------------------------------------

log_diff_vars = [
    "m0_eom",
    "m1_eom",
    "m2_eom",
    "m2x_eom",
    "usd_rub_avg",
    "brent_avg",
    "nominal_wage",
    "reserves_eom",
    "ffpi_food"
]

for var in log_diff_vars:
    if var in df.columns:
        df[f"dlog_{var}"] = np.log(df[var]).diff()

# ---------------------------------------------------------
# 4. сезонные фиктивные переменные
# ---------------------------------------------------------

df["month"] = df.index.month
month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True)
df = pd.concat([df, month_dummies], axis=1)

# ---------------------------------------------------------
# 5. удаление служебных колонок
# ---------------------------------------------------------

df = df.drop(columns=["month"], errors="ignore")

# ---------------------------------------------------------
# 6. удаление первых строк с лагами
# ---------------------------------------------------------

df = df.dropna(subset=["cpi_lag1", "cpi_lag2", "cpi_lag3"])

# ---------------------------------------------------------
# 7. сохранение
# ---------------------------------------------------------

df.to_csv("dataset_for_model_rich.csv")
df.to_excel("dataset_for_model_rich.xlsx")

print("Dataset shape:", df.shape)