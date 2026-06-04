import pandas as pd
import matplotlib.pyplot as plt

# 1. Загрузка данных
df = pd.read_csv(
    "tvp_direct_hybrid_ml_dominant_with_shap_h12.csv",
    sep=";",
    decimal=","
)
print(df)
# 2. Преобразование даты
df["forecast_origin"] = pd.to_datetime(df["forecast_origin"], format="%d.%m.%Y")

# 3. SHAP
df["abs_shap_value"] = df["shap_value"].abs()
print(df)
# 4. Выбор конкретной модели и варианта
base_model = "TVP"
variant = "linear_meta"

plot_df = df[
    (df["base_model"] == base_model) &
    (df["variant"] == variant)
].copy()

# 5. Сортировка
plot_df = plot_df.sort_values(["feature", "forecast_origin"])
pd.set_option('display.max_rows', None)
print(plot_df)

# 6. Построение графика
plt.figure(figsize=(14, 7))

for feature, group in plot_df.groupby("feature"):
    plt.plot(
        group["forecast_origin"],
        group["abs_shap_value"],
        marker="o",
        label=feature
    )

plt.title(f"Изменение abs SHAP-value во времени\nbase_model={base_model}, variant={variant}")
plt.xlabel("Дата")
plt.ylabel("abs SHAP-value")
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
