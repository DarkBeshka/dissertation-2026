import pandas as pd
import numpy as np

# ---------------------------------------------------------
# 1. загрузка длинной мастер-таблицы
# ---------------------------------------------------------
# Ожидаемый входной файл: master_table_russia_2001_01_long_sample.csv
# Столбец даты: date

INPUT_CSV = "master_table_russia_2001_01_long_sample.csv"
OUTPUT_CSV = "dataset_for_model_long.csv"
OUTPUT_XLSX = "dataset_for_model_long.xlsx"


def safe_log_diff(series: pd.Series) -> pd.Series:
    """Вычисляет log-diff только для строго положительных значений."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.where(s > 0)
    return np.log(s).diff()


def main() -> None:
    df = pd.read_csv(INPUT_CSV, parse_dates=["date"])
    df = df.sort_values("date").set_index("date")

    # -----------------------------------------------------
    # 2. лаги инфляции
    # -----------------------------------------------------
    if "cpi_mm" not in df.columns:
        raise KeyError("В мастер-таблице отсутствует обязательный столбец 'cpi_mm'.")

    df["cpi_lag1"] = df["cpi_mm"].shift(1)
    df["cpi_lag2"] = df["cpi_mm"].shift(2)
    df["cpi_lag3"] = df["cpi_mm"].shift(3)

    # -----------------------------------------------------
    # 3. лог-разности для уровневых рядов длинной выборки
    # -----------------------------------------------------
    # Используем только те переменные, которые обсуждались для long sample.
    log_diff_vars = [
        "m2_eom",
        "usd_rub_avg",
        "brent_avg",
        "nominal_wage",
        "ffpi_food",
    ]

    for var in log_diff_vars:
        if var in df.columns:
            df[f"dlog_{var}"] = safe_log_diff(df[var])

    # Дополнительно оставляем уже существующие трансформации из мастер-таблицы, которые были посчитаны на этапе сборки.
    # Например: m2_log_mom, m2_log_yoy, wage_log_yoy, ffpi_food_log_mom, usd_rub_log_mom, brent_log_mom.

    # -----------------------------------------------------
    # 4. сезонные фиктивные переменные
    # -----------------------------------------------------
    df["month"] = df.index.month
    month_dummies = pd.get_dummies(df["month"], prefix="month", drop_first=True, dtype=int)
    df = pd.concat([df, month_dummies], axis=1)

    # -----------------------------------------------------
    # 5. удаление служебной колонки
    # -----------------------------------------------------
    df = df.drop(columns=["month"], errors="ignore")

    # -----------------------------------------------------
    # 6. удаление первых строк, где не хватает лагов таргета
    # -----------------------------------------------------
    df = df.dropna(subset=["cpi_lag1", "cpi_lag2", "cpi_lag3"])

    # -----------------------------------------------------
    # 7. сохранение
    # -----------------------------------------------------
    df.to_csv(OUTPUT_CSV, encoding="utf-8-sig")
    df.to_excel(OUTPUT_XLSX)

    print("Dataset shape:", df.shape)
    print("Saved:", OUTPUT_CSV)
    print("Saved:", OUTPUT_XLSX)


if __name__ == "__main__":
    main()
