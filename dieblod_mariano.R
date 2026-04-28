# Установка и подключение пакета
# install.packages("forecast")
library(forecast)

# Загрузка данных
ar3_res <- read.csv("ран который в дипломе/fixed_ar3_residual_hybrid_unrestricted_test_h12.csv",     sep = ";",
                    dec = ",",
                    stringsAsFactors = FALSE,
                    check.names = FALSE,   fileEncoding = "CP1251")
base    <- read.csv("ран который в дипломе/fixed_base_test_results_h12.csv",     sep = ",")
ar3_dir <- read.csv("ран который в дипломе/fixed_tvp_direct_hybrid_linear_meta_test_h12.csv",     sep = ";", dec=",",
                    stringsAsFactors = FALSE,
                    check.names = FALSE,   fileEncoding = "CP1251")
xgb     <- read.csv("ран который в дипломе/fixed_xgb_direct_base_test_h12.csv", fileEncoding = "CP1251")

# Фактические значения
y <- ar3_res$actual

# Прогнозы моделей
fc_ar3_res <- ar3_res$forecast_hybrid_residual
fc_ar3_dir <- ar3_dir$forecast_hybrid
fc_ar3     <- base$forecast_ar3
fc_ols     <- base$forecast_ols
fc_tvp     <- base$forecast_tvp
fc_xgb     <- xgb$forecast_xgb

# Функция для безопасного DM-теста (убираем NA)
run_dm <- function(f1, f2, y) {
  idx <- complete.cases(f1, f2, y)
  dm.test(
    e1 = y[idx] - f1[idx],
    e2 = y[idx] - f2[idx],
    alternative = "two.sided",
    h = 1,
    power = 2   # квадратичная функция потерь (MSE)
  )
}

# --- 1. AR3 + residual_hybrid против остальных ---
dm_res_vs_all <- list(
  AR3   = run_dm(fc_ar3_res, fc_ar3, y),
  OLS   = run_dm(fc_ar3_res, fc_ols, y),
  TVP   = run_dm(fc_ar3_res, fc_tvp, y),
  XGB   = run_dm(fc_ar3_res, fc_xgb, y),
  AR3_direct = run_dm(fc_ar3_res, fc_ar3_dir, y)
)

# --- 2. AR3 + direct_hybrid против остальных ---
dm_dir_vs_all <- list(
  AR3   = run_dm(fc_ar3_dir, fc_ar3, y),
  OLS   = run_dm(fc_ar3_dir, fc_ols, y),
  TVP   = run_dm(fc_ar3_dir, fc_tvp, y),
  XGB   = run_dm(fc_ar3_dir, fc_xgb, y),
  AR3_residual = run_dm(fc_ar3_dir, fc_ar3_res, y)
)

# Преобразование в таблицу
extract_results <- function(dm_list, base_name) {
  data.frame(
    Model_1 = base_name,
    Model_2 = names(dm_list),
    DM_stat = sapply(dm_list, function(x) x$statistic),
    p_value = sapply(dm_list, function(x) x$p.value)
  )
}

results_table <- rbind(
  extract_results(dm_res_vs_all, "AR3_residual_hybrid"),
  extract_results(dm_dir_vs_all, "AR3_direct_hybrid")
)

print(results_table)