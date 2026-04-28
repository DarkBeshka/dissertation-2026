# import pandas as pd
# import matplotlib.pyplot as plt

# # === 1. Загрузка данных ===
# file_path = "fixed train/fixed_direct_hybrid_ml_dominant_with_shap_h1.csv"
# df = pd.read_csv(file_path)

# # === 2. Фильтрация нужного варианта ===
# df = df[df["variant"] == "linear_meta"].copy()
# df = df[df["base_model"] == "TVP"].copy()

# # === 3. Подготовка данных ===
# df["forecast_origin"] = pd.to_datetime(df["forecast_origin"])

# # SHAP-важность = модуль значения
# df["importance"] = df["shap_value"].abs()

# # Агрегация (если есть дубликаты по дате/фиче)
# plot_df = (
#     df.groupby(["forecast_origin", "feature"], as_index=False)["importance"]
#       .mean()
# )

# # === 4. Pivot ===
# pivot_df = plot_df.pivot(
#     index="forecast_origin",
#     columns="feature",
#     values="importance"
# ).sort_index()

# # === 5. График ===
# plt.figure(figsize=(14, 8))

# for feature in pivot_df.columns:
#     plt.plot(
#         pivot_df.index,
#         pivot_df[feature],
#         label=feature,
#         linewidth=1.5
#     )

# plt.title("Динамика важности предикторов")
# plt.xlabel("Дата")
# plt.ylabel("|SHAP|")
# plt.grid(True, alpha=0.3)
# plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
# plt.tight_layout()
# plt.show()

# import pandas as pd
# import matplotlib.pyplot as plt

# # === 1. Загрузка данных ===
# file_path = "fixed_direct_hybrid_ml_dominant_with_shap_h12.csv"
# df = pd.read_csv(file_path, sep=";", decimal=",")

# print(df)

# # === 2. Фильтрация нужного варианта ===
# df = df[df["variant"] == "linear_meta"].copy()
# df = df[df["base_model"] == "AR3"].copy()
# print(df)
# # Если нужен конкретный горизонт, можно раскомментировать:
# # df = df[df["horizon"] == 1].copy()

# # === 3. Подготовка данных ===
# df["forecast_origin"] = pd.to_datetime(df["forecast_origin"])

# # Находим все колонки с SHAP-значениями
# shap_cols = [col for col in df.columns if col.startswith("shap_")]
# # shap_cols.append("BIAS")

# # Преобразуем из wide -> long:
# # каждая строка станет парой (дата, предиктор, shap_value)
# plot_df = df.melt(
#     id_vars=["forecast_origin"],
#     value_vars=shap_cols,
#     var_name="feature",
#     value_name="shap_value"
# )

# # Убираем префикс shap_ из названия предиктора
# plot_df["feature"] = plot_df["feature"].str.replace("shap_", "", regex=False)

# # SHAP-важность = модуль значения
# plot_df["importance"] = plot_df["shap_value"].abs()

# # Агрегация на случай дублей по дате/фиче
# # plot_df = (
# #     plot_df.groupby(["forecast_origin", "feature"], as_index=False)["importance"]
# #            .mean()
# # )

# # === 4. Pivot ===
# pivot_df = plot_df.pivot(
#     index="forecast_origin",
#     columns="feature",
#     values="importance"
# ).sort_index()

# # === 5. График ===
# plt.figure(figsize=(14, 8))

# for feature in pivot_df.columns:
#     plt.plot(
#         pivot_df.index,
#         pivot_df[feature],
#         label=feature,
#         linewidth=1.5
#     )

# plt.title("Динамика важности предикторов")
# plt.xlabel("Дата")
# plt.ylabel("|SHAP|")
# plt.grid(True, alpha=0.3)
# plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
# plt.tight_layout()
# plt.show()
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

# 3. Модуль SHAP
df["abs_shap_value"] = df["shap_value"].abs()
print(df)
# 4. При необходимости выберите конкретную модель и variant
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