library(tidyverse)
library(lubridate)
library(shrinkTVP)
library(coda)
library(xgboost)
library(forecast)
library(iml)
library(ggplot2)

# =========================================================
# 0. SETTINGS
# =========================================================
train_start <- as.Date("2001-01-01")
train_end   <- as.Date("2023-12-01")
test_start  <- as.Date("2024-01-01")

horizons <- c(1, 3, 6, 12)

# TVP
niter_tvp <- 15000
nburn_tvp <- 7000

tvp_predictors <- c(
  "cpi_lag1", "cpi_lag2", "cpi_lag3", #"cpi_seasdiff_12",
  "usd_rub_log_mom",
  "business_climate_cbr",
  "unemployment_rate",
  paste0("month_", 2:12)
)

ml_predictors <- c(
  "cpi_lag1", "cpi_lag2", "cpi_lag3",# "cpi_seasdiff_12",
  "m2_log_mom",
  "usd_rub_log_mom",
  "brent_log_mom",
  "business_climate_cbr",
  "unemployment_rate",
  "ppi_construction_mm",
  "ffpi_food_log_mom",
  paste0("month_", 2:12)
  
)

ols_predictors <- tvp_predictors

ar3_formula <- y_h ~ cpi_lag1 + cpi_lag2 + cpi_lag3
ols_formula <- as.formula(
  paste("y_h ~", paste(ols_predictors, collapse = " + "))
)

# XGB / residual hybrids
min_ml_obs <- 48
xgb_nrounds <- 150
xgb_eta <- 0.03
xgb_subsample <- 0.8
xgb_colsample_bytree <- 0.8
xgb_max_depth_default <- 3L
xgb_max_depth_restricted <- 2L
recency_decay_lambda <- 0.8
ml_window_years <- 10

hybrid_variants <- c("last10y", "weighted", "shallow", "unrestricted") #
hybrid_base_models <- c("TVP", "AR3", "OLS")

# direct XGB
min_xgb_obs <- 48

# direct hybrids
direct_hybrid_variants <- c(
  "fixed_6040",
  "rolling_rmse",
  "linear_meta",
  "regime_threshold"
  # "anchor_adjustment"
)

fixed_weight_base_h1 <- 0.60
fixed_weight_base_other <- 0.60

rolling_window_months <- 12
rolling_weight_lower <- 0.20
rolling_weight_upper <- 0.80

linear_meta_min_obs <- 24

regime_inflation_threshold <- 1.0
regime_weight_base_normal <- 0.70
regime_weight_base_stress <- 0.30

anchor_window_months <- 12
anchor_gamma_lower <- 0.00
anchor_gamma_upper <- 1.00

# XAI: ALE / Interaction
ale_grid_size <- 20L
interaction_top_n <- 15L
interaction_sample_size <- 200L
interaction_seed <- 123

# XAI plots
xai_plot_dir <- "xai_plots"
dir.create(xai_plot_dir, showWarnings = FALSE, recursive = TRUE)

shock_periods <- tibble::tribble(
  ~start,          ~end,
  # as.Date("2008-09-01"), as.Date("2009-03-01"),
  as.Date("2014-12-01"), as.Date("2015-03-01"),
  # as.Date("2020-03-01"), as.Date("2020-04-01"),
  as.Date("2022-01-01"), as.Date("2022-06-01")
)

max_lag_months <- 12
max_horizon_months <- max(horizons)

# =========================================================
# 0.1 MANUAL XGB PARAMS
# =========================================================

xgb_direct_params_tbl <- tibble::tribble(
  ~horizon, ~max_depth, ~nrounds, ~eta,   ~subsample, ~colsample_bytree, ~min_child_weight, ~gamma, ~lambda, ~alpha, ~max_delta_step,
  1L,       2L,         160L,     0.01,  1.0,        0.5,               15,                1.0,    10,       0.1,    0,
  3L,       2L,        1000L,     0.005, 1.0,        0.8,               10,                1.2,    60,       3.0,    3,
  6L,       2L,         700L,     0.008, 1.0,        0.8,               15,                1.0,    40,       6.0,    3,
  12L,       2L,         700L,     0.008, 1.0,        0.8,               15,                1.0,    40,       6.0,    3
)

residual_xgb_params_tbl <- tibble::tribble(
  ~horizon, ~base_model, ~variant,         ~max_depth_default, ~max_depth_restricted, ~nrounds, ~eta,  ~subsample, ~colsample_bytree, ~min_child_weight, ~gamma, ~lambda, ~alpha, ~max_delta_step,
  
  1L,      "OLS",       "unrestricted",    2L,                 2L,                    400L,     0.02,  1.0,        0.6,               5,                 0.2,    1,     1.0,    0,
  1L,      "AR3",       "unrestricted",    2L,                 2L,                    400L,     0.02,  1.0,        0.6,               5,                 0.2,    5,     1.0,    0,
  1L,      "TVP",       "unrestricted",    2L,                 2L,                    450L,     0.02,  1.0,        0.6,               5,                 0.4,    1,     1.0,    0,
  
  3L,      "AR3",       "unrestricted",    2L,                 2L,                    400L,     0.01,  1.0,        0.7,              15,                 1.0,   10.0,     1.0,    3,
  3L,      "OLS",       "unrestricted",    2L,                 2L,                    400L,     0.02,  1.0,        0.4,               5,                 0.2,    0.1,     0.0,    0,
  3L,      "TVP",       "unrestricted",    2L,                 2L,                    400L,     0.03,  1.0,        0.6,               5,                 0.2,    0.5,     0.0,    0,
  
  6L,      "TVP",       "unrestricted",    2L,                 2L,                    400L,     0.01,  1.0,        0.7,              15,                 1.7,  300.0,     4.0,    3,
  6L,      "AR3",       "unrestricted",    2L,                 2L,                    400L,     0.05,  1.0,        0.9,              15,                 0.5,  200.0,     5.0,    3,
  6L,      "OLS",       "unrestricted",    2L,                 2L,                    400L,     0.05,  1.0,        0.9,              15,                 0.5,  200.0,     5.0,    3,
  
  12L,      "TVP",       "unrestricted",    2L,                 2L,                    400L,     0.05,  1.0,        0.3,              10,                 0.1,  200.0,     1.0,    0,
  12L,      "OLS",       "unrestricted",    2L,                 2L,                    400L,     0.05,  1.0,        0.3,              10,                 0.1,  200.0,     1.0,    0,
  12L,      "AR3",       "shallow",         2L,                 2L,                    800L,     0.03,  1.0,        0.5,              10,                 0.7,   70.0,     0.7,    3,
  12L,      "AR3",       "unrestricted",    2L,                 2L,                    800L,     0.03,  1.0,        0.5,              10,                 0.7,   70.0,     0.7,    3
)

get_xgb_direct_params <- function(h) {
  row <- xgb_direct_params_tbl %>% filter(horizon == h)
  
  if (nrow(row) == 0) {
    return(list(
      max_depth = xgb_max_depth_default,
      nrounds = xgb_nrounds,
      eta = xgb_eta,
      subsample = xgb_subsample,
      colsample_bytree = xgb_colsample_bytree,
      min_child_weight = 1,
      gamma = 0,
      lambda = 1,
      alpha = 0,
      max_delta_step = 0
    ))
  }
  
  as.list(row[1, c(
    "max_depth", "nrounds", "eta", "subsample", "colsample_bytree",
    "min_child_weight", "gamma", "lambda", "alpha", "max_delta_step"
  )])
}

get_residual_xgb_params <- function(h, base_model, variant) {
  row <- residual_xgb_params_tbl %>%
    filter(horizon == h, base_model == !!base_model, variant == !!variant)
  
  if (nrow(row) == 0) {
    return(list(
      max_depth_default = xgb_max_depth_default,
      max_depth_restricted = xgb_max_depth_restricted,
      nrounds = xgb_nrounds,
      eta = xgb_eta,
      subsample = xgb_subsample,
      colsample_bytree = xgb_colsample_bytree,
      min_child_weight = 1,
      gamma = 0,
      lambda = 1,
      alpha = 0,
      max_delta_step = 0
    ))
  }
  
  as.list(row[1, c(
    "max_depth_default", "max_depth_restricted",
    "nrounds", "eta", "subsample", "colsample_bytree",
    "min_child_weight", "gamma", "lambda", "alpha", "max_delta_step"
  )])
}

# =========================================================
# 1. DATA
# =========================================================
data <- read_csv("dataset_for_model_long.csv", show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

# data <- data %>%
#   arrange(date) %>%
#   mutate(
#     cpi_lag12 = lag(cpi_mm, 12),
#     cpi_seasdiff_12 = cpi_mm - lag(cpi_mm, 12)
#   )

data <- data %>%
  arrange(date) %>%
  mutate(
    month_num = lubridate::month(date),
    cpi_seasdiff_12 = cpi_mm - lag(cpi_mm, 12)
  )

data_rich <- read_csv("dataset_for_model_rich.csv", show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

rich_predictors <- setdiff(
  names(data_rich),
  c("date", "cpi_mm", "y_h")
)

shock_periods_expanded <- shock_periods %>%
  mutate(
    purge_start = start %m-% months(max_horizon_months),
    purge_end   = end   %m+% months(max_lag_months)
  )

remove_shock_periods <- function(df, shock_tbl) {
  if (nrow(shock_tbl) == 0) return(df)
  
  keep_idx <- rep(TRUE, nrow(df))
  
  for (i in seq_len(nrow(shock_tbl))) {
    keep_idx <- keep_idx & !(
      df$date >= shock_tbl$purge_start[i] &
        df$date <= shock_tbl$purge_end[i]
    )
  }
  
  df[keep_idx, , drop = FALSE]
}

# потом вырезаем шоковые периоды вместе с защитным буфером
data <- remove_shock_periods(data, shock_periods_expanded) %>%
  arrange(date)

# =========================================================
# 2. HELPERS
# =========================================================
safe_mean <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  mean(x, na.rm = TRUE)
}

safe_median <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA_real_)
  median(x, na.rm = TRUE)
}

safe_sd <- function(x) {
  if (length(x) <= 1 || all(is.na(x))) return(NA_real_)
  sd(x, na.rm = TRUE)
}

make_horizon_target <- function(df, h) {
  df %>%
    arrange(date) %>%
    mutate(y_h = lead(cpi_mm, h))
}

compute_decay_weights <- function(n, lambda = recency_decay_lambda) {
  idx <- seq_len(n)
  lambda^(rev(idx) - 1)
}

clamp_value <- function(x, lower, upper) {
  pmax(lower, pmin(upper, x))
}

compute_backtest_metrics <- function(results_df) {
  results_df %>%
    summarise(
      n_forecasts_tvp = sum(!is.na(forecast_tvp) & !is.na(actual)),
      mae_tvp = mean(abs(actual - forecast_tvp), na.rm = TRUE),
      rmse_tvp = sqrt(mean((actual - forecast_tvp)^2, na.rm = TRUE)),
      bias_tvp = mean(forecast_tvp - actual, na.rm = TRUE),
      
      n_forecasts_ols = sum(!is.na(forecast_ols) & !is.na(actual)),
      mae_ols = mean(abs(actual - forecast_ols), na.rm = TRUE),
      rmse_ols = sqrt(mean((actual - forecast_ols)^2, na.rm = TRUE)),
      bias_ols = mean(forecast_ols - actual, na.rm = TRUE),
      
      n_forecasts_ar3 = sum(!is.na(forecast_ar3) & !is.na(actual)),
      mae_ar3 = mean(abs(actual - forecast_ar3), na.rm = TRUE),
      rmse_ar3 = sqrt(mean((actual - forecast_ar3)^2, na.rm = TRUE)),
      bias_ar3 = mean(forecast_ar3 - actual, na.rm = TRUE),
      
      n_forecasts_arima = sum(!is.na(forecast_arima) & !is.na(actual)),
      mae_arima = mean(abs(actual - forecast_arima), na.rm = TRUE),
      rmse_arima = sqrt(mean((actual - forecast_arima)^2, na.rm = TRUE)),
      bias_arima = mean(forecast_arima - actual, na.rm = TRUE)
    )
}

compute_xgb_direct_metrics <- function(df, model_name, h) {
  tibble(
    horizon = h,
    model = model_name,
    n_forecasts = sum(!is.na(df$forecast_xgb) & !is.na(df$actual)),
    mae = mean(abs(df$actual - df$forecast_xgb), na.rm = TRUE),
    rmse = sqrt(mean((df$actual - df$forecast_xgb)^2, na.rm = TRUE)),
    bias = mean(df$forecast_xgb - df$actual, na.rm = TRUE)
  )
}

compute_hybrid_metrics <- function(df, base_model, variant_name, h) {
  tibble(
    horizon = h,
    base_model = base_model,
    variant = variant_name,
    
    n_base = sum(!is.na(df$base_forecast) & !is.na(df$actual)),
    mae_base = mean(abs(df$actual - df$base_forecast), na.rm = TRUE),
    rmse_base = sqrt(mean((df$actual - df$base_forecast)^2, na.rm = TRUE)),
    bias_base = mean(df$base_forecast - df$actual, na.rm = TRUE),
    
    n_hybrid = sum(!is.na(df$forecast_hybrid) & !is.na(df$actual)),
    mae_hybrid = mean(abs(df$actual - df$forecast_hybrid), na.rm = TRUE),
    rmse_hybrid = sqrt(mean((df$actual - df$forecast_hybrid)^2, na.rm = TRUE)),
    bias_hybrid = mean(df$forecast_hybrid - df$actual, na.rm = TRUE)
  )
}

compute_direct_hybrid_metrics <- function(df, base_model, variant_name, h) {
  tibble(
    horizon = h,
    base_model = base_model,
    variant = variant_name,
    
    n_base = sum(!is.na(df$base_forecast) & !is.na(df$actual)),
    mae_base = mean(abs(df$actual - df$base_forecast), na.rm = TRUE),
    rmse_base = sqrt(mean((df$actual - df$base_forecast)^2, na.rm = TRUE)),
    bias_base = mean(df$base_forecast - df$actual, na.rm = TRUE),
    
    n_ml = sum(!is.na(df$ml_forecast) & !is.na(df$actual)),
    mae_ml = mean(abs(df$actual - df$ml_forecast), na.rm = TRUE),
    rmse_ml = sqrt(mean((df$actual - df$ml_forecast)^2, na.rm = TRUE)),
    bias_ml = mean(df$ml_forecast - df$actual, na.rm = TRUE),
    
    n_hybrid = sum(!is.na(df$forecast_hybrid) & !is.na(df$actual)),
    mae_hybrid = mean(abs(df$actual - df$forecast_hybrid), na.rm = TRUE),
    rmse_hybrid = sqrt(mean((df$actual - df$forecast_hybrid)^2, na.rm = TRUE)),
    bias_hybrid = mean(df$forecast_hybrid - df$actual, na.rm = TRUE)
  )
}

sanitize_filename <- function(x) {
  x %>%
    stringr::str_replace_all("[^A-Za-z0-9_\\-]+", "_") %>%
    stringr::str_replace_all("_+", "_") %>%
    stringr::str_replace_all("^_|_$", "")
}

safe_dir_create <- function(path) {
  if (!dir.exists(path)) dir.create(path, recursive = TRUE, showWarnings = FALSE)
}

# =========================================================
# 3. TVP HELPERS
# =========================================================
extract_predictive_draws <- function(forc_obj, horizon = 1) {
  if (!inherits(forc_obj, "shrinkTVP_forc")) {
    stop("forc_obj must inherit from class 'shrinkTVP_forc'")
  }
  if (is.null(forc_obj$y_pred)) {
    stop("shrinkTVP_forc object has no 'y_pred' field")
  }
  
  y_pred <- forc_obj$y_pred
  if (!(is.data.frame(y_pred) || is.matrix(y_pred))) {
    y_pred <- tryCatch(as.data.frame(y_pred), error = function(e) NULL)
    if (is.null(y_pred)) {
      stop("'y_pred' cannot be coerced to data.frame")
    }
  }
  if (ncol(y_pred) < horizon) {
    stop(sprintf("'y_pred' has only %d col(s), horizon=%d", ncol(y_pred), horizon))
  }
  
  draws <- y_pred[[horizon]]
  draws <- as.numeric(draws)
  draws <- draws[is.finite(draws)]
  
  if (length(draws) == 0) {
    stop("No finite posterior predictive draws found in y_pred")
  }
  draws
}

extract_point_forecast <- function(forc_obj, horizon = 1, stat = c("median", "mean")) {
  stat <- match.arg(stat)
  draws <- extract_predictive_draws(forc_obj, horizon = horizon)
  if (stat == "median") median(draws, na.rm = TRUE) else mean(draws, na.rm = TRUE)
}

extract_forecast_summary <- function(forc_obj, horizon = 1,
                                     probs = c(0.025, 0.5, 0.975)) {
  draws <- extract_predictive_draws(forc_obj, horizon = horizon)
  qs <- quantile(draws, probs = probs, na.rm = TRUE, names = FALSE)
  
  tibble(
    forecast_mean = mean(draws, na.rm = TRUE),
    forecast_median = median(draws, na.rm = TRUE),
    forecast_sd = sd(draws, na.rm = TRUE),
    forecast_q025 = qs[1],
    forecast_q50 = qs[2],
    forecast_q975 = qs[3],
    n_draws = length(draws)
  )
}

to_draw_matrix <- function(x) {
  m <- tryCatch(as.matrix(x), error = function(e) NULL)
  if (is.null(m)) return(NULL)
  m <- as.matrix(m)
  if (!is.numeric(m)) storage.mode(m) <- "double"
  m
}

normalize_beta_list <- function(beta_obj, predictor_names, fit_colnames = NULL) {
  if (!is.list(beta_obj)) {
    stop("fit$beta is not a list")
  }
  
  beta_list <- beta_obj
  
  beta_names <- names(beta_list)
  if (is.null(beta_names) || any(beta_names == "")) {
    stop("fit$beta has no usable names")
  }
  
  # Нормализуем имена коэффициентов shrinkTVP:
  # beta_cpi_lag1 -> cpi_lag1
  # beta_Intercept -> Intercept
  beta_names <- gsub("^beta_", "", beta_names)
  beta_names <- gsub("^\\(Intercept\\)$", "Intercept", beta_names)
  names(beta_list) <- beta_names
  
  # Если fit_colnames есть, можно использовать его для диагностики,
  # но не делать обязательным.
  if (!is.null(fit_colnames)) {
    fit_colnames <- gsub("^\\(Intercept\\)$", "Intercept", fit_colnames)
  }
  
  missing_preds <- setdiff(predictor_names, names(beta_list))
  if (length(missing_preds) > 0) {
    stop(
      paste(
        "Missing predictors in fit$beta after name normalization:",
        paste(missing_preds, collapse = ", ")
      )
    )
  }
  
  beta_list
}

extract_last_beta_state <- function(beta_mcmc_tvp) {
  m <- to_draw_matrix(beta_mcmc_tvp)
  if (is.null(m)) {
    return(tibble(
      beta_last_mean = NA_real_,
      beta_last_q025 = NA_real_,
      beta_last_q975 = NA_real_,
      beta_last_nonzero_95 = NA
    ))
  }
  
  last_draws <- m[, ncol(m)]
  q025 <- quantile(last_draws, probs = 0.025, na.rm = TRUE)
  q975 <- quantile(last_draws, probs = 0.975, na.rm = TRUE)
  
  tibble(
    beta_last_mean = mean(last_draws, na.rm = TRUE),
    beta_last_q025 = q025,
    beta_last_q975 = q975,
    beta_last_nonzero_95 = !(q025 <= 0 && q975 >= 0)
  )
}

compute_tvp_global_contribution <- function(beta_last_df, h) {
  if (is.null(beta_last_df) || nrow(beta_last_df) == 0) {
    return(tibble(
      horizon = integer(),
      predictor = character(),
      mean_abs_beta_last = numeric(),
      mean_beta_last = numeric(),
      sd_beta_last = numeric(),
      share_nonzero_95 = numeric(),
      n = integer()
    ))
  }
  
  beta_last_df %>%
    group_by(predictor) %>%
    summarise(
      mean_abs_beta_last = mean(abs(beta_last_mean), na.rm = TRUE),
      mean_beta_last = mean(beta_last_mean, na.rm = TRUE),
      sd_beta_last = sd(beta_last_mean, na.rm = TRUE),
      share_nonzero_95 = mean(beta_last_nonzero_95, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    mutate(horizon = h) %>%
    select(
      horizon, predictor,
      mean_abs_beta_last, mean_beta_last,
      sd_beta_last, share_nonzero_95, n
    ) %>%
    arrange(horizon, desc(mean_abs_beta_last))
}

# =========================================================
# 4. BASE MODEL FIT/PREDICT
# =========================================================
fit_lm_direct <- function(train_df, formula_obj) {
  train_cc <- model.frame(formula_obj, data = train_df, na.action = na.omit)
  if (nrow(train_cc) < 24) return(NULL)
  lm(formula_obj, data = train_df, na.action = na.omit)
}

extract_lm_coef_table <- function(fit, h, model_name) {
  if (is.null(fit)) {
    return(tibble(
      horizon = h,
      model = model_name,
      term = NA_character_,
      estimate = NA_real_,
      std_error = NA_real_,
      t_value = NA_real_,
      p_value = NA_real_
    ))
  }
  
  coef_mat <- summary(fit)$coefficients
  
  tibble(
    horizon = h,
    model = model_name,
    term = rownames(coef_mat),
    estimate = coef_mat[, 1],
    std_error = coef_mat[, 2],
    t_value = coef_mat[, 3],
    p_value = coef_mat[, 4]
  )
}

predict_lm_fixed <- function(fit, full_df, forecast_col_name) {
  out <- full_df %>% select(date, y_h) %>% mutate(pred = NA_real_)
  if (is.null(fit)) {
    return(out %>% rename(!!forecast_col_name := pred, actual = y_h, forecast_origin = date))
  }
  
  pred <- tryCatch(
    as.numeric(predict(fit, newdata = full_df)),
    error = function(e) rep(NA_real_, nrow(full_df))
  )
  
  out$pred <- pred
  out %>% rename(!!forecast_col_name := pred, actual = y_h, forecast_origin = date)
}

fit_arima_direct <- function(train_df) {
  train_cc <- train_df %>%
    select(y_h) %>%
    drop_na()
  
  if (nrow(train_cc) < 24) return(NULL)
  
  tryCatch(
    auto.arima(train_cc$y_h),
    error = function(e) NULL
  )
}

predict_arima_fixed <- function(fit, test_df) {
  # strict fixed-model:
  # one model on train, then one multi-step forecast over entire test block
  out <- test_df %>% select(date, y_h) %>% mutate(pred = NA_real_)
  if (is.null(fit) || nrow(test_df) == 0) {
    return(out %>% rename(forecast_arima = pred, actual = y_h, forecast_origin = date))
  }
  
  fc <- tryCatch(
    forecast::forecast(fit, h = nrow(test_df)),
    error = function(e) NULL
  )
  
  if (!is.null(fc)) {
    out$pred <- as.numeric(fc$mean)
  }
  
  out %>% rename(forecast_arima = pred, actual = y_h, forecast_origin = date)
}

fit_tvp_direct <- function(train_df, predictors, niter = niter_tvp, nburn = nburn_tvp) {
  train_cc <- train_df %>%
    select(all_of(c("y_h", predictors))) %>%
    drop_na()
  
  if (nrow(train_cc) < 24) {
    return(NULL)
  }
  
  tvp_formula <- as.formula(
    paste("y_h ~", paste(predictors, collapse = " + "))
  )
  
  fit <- tryCatch(
    shrinkTVP(
      formula = tvp_formula,
      data = train_cc,
      niter = niter,
      nburn = nburn
    ),
    error = function(e) {
      message("TVP estimation failed: ", conditionMessage(e))
      NULL
    }
  )
  
  fit
}

extract_tvp_beta_last <- function(fit, predictors, forecast_dates) {
  if (is.null(fit) || is.null(fit$beta)) return(tibble())
  
  fit_colnames <- tryCatch(colnames(fit$X), error = function(e) NULL)
  beta_list <- tryCatch(
    normalize_beta_list(
      beta_obj = fit$beta,
      predictor_names = predictors,
      fit_colnames = fit_colnames
    ),
    error = function(e) {
      message("extract_tvp_beta_last: normalize_beta_list failed: ", conditionMessage(e))
      NULL
    }
  )
  
  if (is.null(beta_list)) return(tibble())
  
  beta_static <- bind_rows(lapply(predictors, function(pr) {
    extract_last_beta_state(beta_list[[pr]]) %>%
      mutate(predictor = pr)
  }))
  
  if (length(forecast_dates) == 0) return(beta_static)
  
  bind_rows(lapply(as.Date(forecast_dates), function(d) {
    beta_static %>%
      mutate(forecast_origin = d)
  })) %>%
    select(forecast_origin, predictor, everything())
}

predict_tvp_fixed <- function(fit, full_df, predictors) {
  out <- full_df %>%
    select(date, y_h) %>%
    mutate(forecast_tvp = NA_real_)
  
  if (is.null(fit) || nrow(full_df) == 0) {
    return(list(
      results = out %>% rename(actual = y_h, forecast_origin = date),
      summary = tibble(),
      beta_last = tibble()
    ))
  }
  
  summ_list <- vector("list", nrow(full_df))
  pred_vec <- rep(NA_real_, nrow(full_df))
  
  for (i in seq_len(nrow(full_df))) {
    newdata <- full_df[i, predictors, drop = FALSE]
    if (any(!is.finite(as.matrix(newdata)))) next
    
    forc <- tryCatch(
      forecast_shrinkTVP(
        mod = fit,
        newdata = newdata,
        n.ahead = 1
      ),
      error = function(e) NULL
    )
    
    if (is.null(forc)) next
    
    pred_vec[i] <- tryCatch(
      extract_point_forecast(forc, horizon = 1, stat = "median"),
      error = function(e) NA_real_
    )
    
    summ_i <- tryCatch(
      extract_forecast_summary(forc, horizon = 1),
      error = function(e) NULL
    )
    
    if (!is.null(summ_i)) {
      summ_list[[i]] <- summ_i %>%
        mutate(forecast_origin = as.Date(full_df$date[i]))
    }
  }
  
  beta_last_df <- extract_tvp_beta_last(fit, predictors, full_df$date)
  
  list(
    results = out %>%
      mutate(forecast_tvp = pred_vec) %>%
      rename(actual = y_h, forecast_origin = date),
    summary = bind_rows(summ_list),
    beta_last = beta_last_df
  )
}

prepare_direct_xgb_data_single <- function(df, predictors, target = "y_h") {
  cols_needed <- unique(c("date", predictors, target))
  cc <- df %>%
    select(all_of(cols_needed)) %>%
    drop_na()
  
  if (nrow(cc) == 0) return(NULL)
  
  list(
    cc = cc,
    x = as.matrix(cc %>% select(all_of(predictors))),
    y = cc[[target]]
  )
}

fit_xgb_direct_fixed <- function(train_df, predictors,
                                 min_obs = min_xgb_obs,
                                 max_depth = xgb_max_depth_default,
                                 nrounds = xgb_nrounds,
                                 eta = xgb_eta,
                                 subsample = xgb_subsample,
                                 colsample_bytree = xgb_colsample_bytree,
                                 min_child_weight = 1,
                                 gamma = 0,
                                 lambda = 1,
                                 alpha = 0,
                                 max_delta_step = 0) {
  prep <- prepare_direct_xgb_data_single(train_df, predictors, target = "y_h")
  if (is.null(prep)) {
    return(list(fit = NULL, train_n = 0, train_cc = tibble()))
  }
  
  if (nrow(prep$x) < min_obs) {
    return(list(fit = NULL, train_n = nrow(prep$x), train_cc = prep$cc))
  }
  
  dtrain <- xgb.DMatrix(data = prep$x, label = prep$y)
  params <- list(
    objective = "reg:squarederror",
    eta = eta,
    max_depth = max_depth,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha,
    max_delta_step = max_delta_step
  )
  
  fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
  
  list(fit = fit, train_n = nrow(prep$x), train_cc = prep$cc)
}

predict_xgb_fixed <- function(fit, full_df, predictors, forecast_col_name = "forecast_xgb") {
  out <- full_df %>% select(date, y_h) %>% mutate(pred = NA_real_)
  
  if (is.null(fit) || nrow(full_df) == 0) {
    return(out %>% rename(!!forecast_col_name := pred, actual = y_h, forecast_origin = date))
  }
  
  valid_idx <- which(stats::complete.cases(full_df[, predictors, drop = FALSE]))
  if (length(valid_idx) > 0) {
    x_test <- as.matrix(full_df[valid_idx, predictors, drop = FALSE])
    dtest <- xgb.DMatrix(data = x_test)
    pred <- as.numeric(predict(fit, dtest))
    out$pred[valid_idx] <- pred
  }
  
  out %>% rename(!!forecast_col_name := pred, actual = y_h, forecast_origin = date)
}

# =========================================================
# 5. RESIDUAL HYBRIDS (ONE FIT ON TRAIN)
# =========================================================
prepare_ml_single_dataset <- function(df, predictors, target = "residual") {
  cols_needed <- unique(c("date", "forecast_origin", predictors, target))
  existing_cols <- intersect(cols_needed, names(df))
  
  cc <- df %>%
    select(all_of(existing_cols)) %>%
    drop_na()
  
  if (nrow(cc) == 0) return(NULL)
  
  list(
    cc = cc,
    x = as.matrix(cc %>% select(all_of(predictors))),
    y = cc[[target]]
  )
}

fit_xgb_residual_fixed <- function(train_df, predictors,
                                   variant = c("last10y", "weighted", "shallow", "unrestricted"),
                                   max_depth_default = xgb_max_depth_default,
                                   max_depth_restricted = xgb_max_depth_restricted,
                                   min_obs = min_ml_obs,
                                   nrounds = xgb_nrounds,
                                   eta = xgb_eta,
                                   subsample = xgb_subsample,
                                   colsample_bytree = xgb_colsample_bytree,
                                   min_child_weight = 1,
                                   gamma = 0,
                                   lambda = 1,
                                   alpha = 0,
                                   max_delta_step = 0) {
  variant <- match.arg(variant)
  
  time_col <- if ("date" %in% names(train_df)) {
    "date"
  } else if ("forecast_origin" %in% names(train_df)) {
    "forecast_origin"
  } else {
    stop("fit_xgb_residual_fixed: neither 'date' nor 'forecast_origin' found")
  }
  
  train_use <- train_df %>% arrange(.data[[time_col]])
  
  if (variant == "last10y") {
    cutoff_date <- max(train_use[[time_col]], na.rm = TRUE) %m-% years(ml_window_years)
    train_use <- train_use %>% filter(.data[[time_col]] >= cutoff_date)
  }
  
  prep <- prepare_ml_single_dataset(train_use, predictors, target = "residual")
  if (is.null(prep)) {
    return(list(fit = NULL, train_n = 0, train_cc = tibble()))
  }
  
  if (nrow(prep$x) < min_obs) {
    return(list(fit = NULL, train_n = nrow(prep$x), train_cc = prep$cc))
  }
  
  weights <- NULL
  max_depth_use <- max_depth_default
  
  if (variant == "weighted") {
    weights <- compute_decay_weights(nrow(prep$x), lambda = recency_decay_lambda)
  }
  
  if (variant == "shallow") {
    max_depth_use <- max_depth_restricted
  }
  
  dtrain <- xgb.DMatrix(data = prep$x, label = prep$y, weight = weights)
  
  params <- list(
    objective = "reg:squarederror",
    eta = eta,
    max_depth = max_depth_use,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha,
    max_delta_step = max_delta_step
  )
  
  fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
  
  list(fit = fit, train_n = nrow(prep$x), train_cc = prep$cc)
}

predict_xgb_residual_fixed <- function(fit, full_df, predictors) {
  out <- full_df %>% mutate(residual_pred_ml = NA_real_)
  
  if (is.null(fit) || nrow(full_df) == 0) {
    return(out)
  }
  
  valid_idx <- which(stats::complete.cases(full_df[, predictors, drop = FALSE]))
  if (length(valid_idx) > 0) {
    x_test <- as.matrix(full_df[valid_idx, predictors, drop = FALSE])
    dtest <- xgb.DMatrix(data = x_test)
    pred <- as.numeric(predict(fit, dtest))
    out$residual_pred_ml[valid_idx] <- pred
  }
  
  out
}

build_hybrid_ml_base <- function(base_results, data, h, base_model = c("TVP", "AR3", "OLS")) {
  base_model <- match.arg(base_model)
  
  df_h <- make_horizon_target(data, h) %>%
    arrange(date)
  
  base_col <- case_when(
    base_model == "TVP" ~ "forecast_tvp",
    base_model == "AR3" ~ "forecast_ar3",
    base_model == "OLS" ~ "forecast_ols"
  )
  
  base_results %>%
    select(forecast_origin, actual, all_of(base_col)) %>%
    rename(base_forecast = all_of(base_col)) %>%
    mutate(
      residual = actual - base_forecast,
      date = forecast_origin
    ) %>%
    left_join(df_h, by = c("forecast_origin" = "date")) %>%
    arrange(forecast_origin)
}

run_residual_hybrid_fixed <- function(train_base_results, test_base_results, data, h,
                                      base_model = c("TVP", "AR3", "OLS"),
                                      variant = c("last10y", "weighted", "shallow", "unrestricted"),
                                      xgb_params = NULL) {
  base_model <- match.arg(base_model)
  variant <- match.arg(variant)
  if (is.null(xgb_params)) {
    xgb_params <- get_residual_xgb_params(h, base_model, variant)
  }
  
  hybrid_train <- build_hybrid_ml_base(train_base_results, data, h, base_model)
  hybrid_test <- build_hybrid_ml_base(test_base_results, data, h, base_model)
  
  fit_ml <- fit_xgb_residual_fixed(
    train_df = hybrid_train,
    predictors = ml_predictors,
    variant = variant,
    max_depth_default = xgb_params$max_depth_default,
    max_depth_restricted = xgb_params$max_depth_restricted,
    nrounds = xgb_params$nrounds,
    eta = xgb_params$eta,
    subsample = xgb_params$subsample,
    colsample_bytree = xgb_params$colsample_bytree,
    min_child_weight = xgb_params$min_child_weight,
    gamma = xgb_params$gamma,
    lambda = xgb_params$lambda,
    alpha = xgb_params$alpha,
    max_delta_step = xgb_params$max_delta_step
  )
  
  hybrid_test_pred <- predict_xgb_residual_fixed(
    fit = fit_ml$fit,
    full_df = hybrid_test,
    predictors = ml_predictors
  ) %>%
    mutate(
      ml_train_n = fit_ml$train_n,
      forecast_hybrid = base_forecast + residual_pred_ml
    )
  
  metrics <- compute_hybrid_metrics(hybrid_test_pred, base_model, variant, h)
  
  list(
    results = hybrid_test_pred,
    metrics = metrics,
    fit = fit_ml$fit,
    train_cc = fit_ml$train_cc,
    hybrid_train_base = hybrid_train,
    hybrid_test_base = hybrid_test
  )
}

run_all_residual_hybrid_fixed <- function(train_base_results, test_base_results, data, h,
                                          base_model = c("TVP", "AR3", "OLS")) {
  base_model <- match.arg(base_model)
  
  variant_runs <- lapply(hybrid_variants, function(v) {
    message("Residual hybrid fixed | base=", base_model, " | h=", h, " | variant=", v)
    
    params_v <- get_residual_xgb_params(h, base_model, v)
    
    run_residual_hybrid_fixed(
      train_base_results = train_base_results,
      test_base_results = test_base_results,
      data = data,
      h = h,
      base_model = base_model,
      variant = v,
      xgb_params = params_v
    )
  })
  names(variant_runs) <- hybrid_variants
  
  metrics_all <- bind_rows(lapply(variant_runs, `[[`, "metrics"))
  
  list(
    runs = variant_runs,
    metrics_all = metrics_all
  )
}

# =========================================================
# 6. DIRECT HYBRIDS (ONE FIT ON TRAIN)
# =========================================================
get_base_forecast_col <- function(base_model = c("TVP", "AR3", "OLS")) {
  base_model <- match.arg(base_model)
  if (base_model == "TVP") return("forecast_tvp")
  if (base_model == "AR3") return("forecast_ar3")
  if (base_model == "OLS") return("forecast_ols")
  stop("Unknown base_model")
}

build_direct_hybrid_base <- function(base_results, xgb_results, data, h,
                                     base_model = c("TVP", "AR3", "OLS")) {
  base_model <- match.arg(base_model)
  base_col <- get_base_forecast_col(base_model)
  
  df_h <- make_horizon_target(data, h) %>%
    arrange(date)
  
  base_results %>%
    select(forecast_origin, actual, all_of(base_col)) %>%
    rename(base_forecast = all_of(base_col)) %>%
    left_join(
      xgb_results %>%
        select(forecast_origin, forecast_xgb) %>%
        rename(ml_forecast = forecast_xgb),
      by = "forecast_origin"
    ) %>%
    left_join(df_h, by = c("forecast_origin" = "date")) %>%
    arrange(forecast_origin)
}

combine_fixed_6040 <- function(base_forecast, ml_forecast, h,
                               w_base_h1 = fixed_weight_base_h1,
                               w_base_other = fixed_weight_base_other) {
  w_base <- ifelse(h == 1, w_base_h1, w_base_other)
  tibble(
    weight_base = w_base,
    weight_ml = 1 - w_base,
    forecast_hybrid = w_base * base_forecast + (1 - w_base) * ml_forecast
  )
}

estimate_static_weight_rmse <- function(train_history,
                                        window_months = rolling_window_months,
                                        weight_lower = rolling_weight_lower,
                                        weight_upper = rolling_weight_upper) {
  hist_use <- train_history %>%
    filter(!is.na(actual), !is.na(base_forecast), !is.na(ml_forecast)) %>%
    arrange(forecast_origin)
  
  if (nrow(hist_use) == 0) return(0.50)
  
  hist_use <- tail(hist_use, window_months)
  
  rmse_base <- sqrt(mean((hist_use$actual - hist_use$base_forecast)^2, na.rm = TRUE))
  rmse_ml <- sqrt(mean((hist_use$actual - hist_use$ml_forecast)^2, na.rm = TRUE))
  
  if (!is.finite(rmse_base) || !is.finite(rmse_ml) || (rmse_base + rmse_ml) <= 0) {
    return(0.50)
  }
  
  w_base <- rmse_ml / (rmse_base + rmse_ml)
  clamp_value(w_base, weight_lower, weight_upper)
}

combine_rolling_rmse_fixed <- function(train_history, test_df,
                                       window_months = rolling_window_months,
                                       weight_lower = rolling_weight_lower,
                                       weight_upper = rolling_weight_upper) {
  w_base <- estimate_static_weight_rmse(
    train_history = train_history,
    window_months = window_months,
    weight_lower = weight_lower,
    weight_upper = weight_upper
  )
  
  test_df %>%
    mutate(
      weight_base = w_base,
      weight_ml = 1 - w_base,
      forecast_hybrid = weight_base * base_forecast + weight_ml * ml_forecast
    )
}

fit_linear_meta_fixed <- function(train_history, min_obs = linear_meta_min_obs) {
  hist_use <- train_history %>%
    filter(!is.na(actual), !is.na(base_forecast), !is.na(ml_forecast))
  
  if (nrow(hist_use) < min_obs) return(NULL)
  
  tryCatch(
    lm(actual ~ base_forecast + ml_forecast, data = hist_use),
    error = function(e) NULL
  )
}

combine_linear_meta_fixed <- function(meta_fit, test_df) {
  if (is.null(meta_fit) || nrow(test_df) == 0) {
    return(
      test_df %>%
        mutate(
          forecast_hybrid = NA_real_,
          meta_intercept = NA_real_,
          meta_coef_base = NA_real_,
          meta_coef_ml = NA_real_
        )
    )
  }
  
  pred <- tryCatch(
    as.numeric(predict(meta_fit, newdata = test_df)),
    error = function(e) rep(NA_real_, nrow(test_df))
  )
  
  cf <- coef(meta_fit)
  intercept <- unname(ifelse("(Intercept)" %in% names(cf), cf["(Intercept)"], NA_real_))
  coef_base <- unname(ifelse("base_forecast" %in% names(cf), cf["base_forecast"], NA_real_))
  coef_ml <- unname(ifelse("ml_forecast" %in% names(cf), cf["ml_forecast"], NA_real_))
  
  test_df %>%
    mutate(
      forecast_hybrid = pred,
      meta_intercept = intercept,
      meta_coef_base = coef_base,
      meta_coef_ml = coef_ml
    )
}

estimate_anchor_gamma_fixed <- function(train_history,
                                        window_months = anchor_window_months,
                                        gamma_lower = anchor_gamma_lower,
                                        gamma_upper = anchor_gamma_upper) {
  hist_use <- train_history %>%
    filter(!is.na(actual), !is.na(base_forecast), !is.na(ml_forecast)) %>%
    arrange(forecast_origin) %>%
    mutate(
      delta_actual = actual - base_forecast,
      delta_model = ml_forecast - base_forecast
    )
  
  if (nrow(hist_use) == 0) return(0.50)
  
  hist_use <- tail(hist_use, window_months)
  
  fit_gamma <- tryCatch(
    lm(delta_actual ~ 0 + delta_model, data = hist_use),
    error = function(e) NULL
  )
  
  if (is.null(fit_gamma)) return(0.50)
  
  cf <- coef(fit_gamma)
  gamma_raw <- unname(ifelse("delta_model" %in% names(cf), cf["delta_model"], 0.50))
  clamp_value(gamma_raw, gamma_lower, gamma_upper)
}

combine_anchor_adjustment_fixed <- function(train_history, test_df,
                                            window_months = anchor_window_months,
                                            gamma_lower = anchor_gamma_lower,
                                            gamma_upper = anchor_gamma_upper) {
  gamma <- estimate_anchor_gamma_fixed(
    train_history = train_history,
    window_months = window_months,
    gamma_lower = gamma_lower,
    gamma_upper = gamma_upper
  )
  
  test_df %>%
    mutate(
      gamma = gamma,
      forecast_hybrid = base_forecast + gamma * (ml_forecast - base_forecast)
    )
}

run_direct_hybrid_fixed <- function(train_base_results,
                                    test_base_results,
                                    xgb_train_results,
                                    xgb_test_results,
                                    data,
                                    h,
                                    base_model = c("TVP", "AR3", "OLS"),
                                    variant = c(
                                      "fixed_6040",
                                      "rolling_rmse",
                                      "linear_meta",
                                      "regime_threshold",
                                      "anchor_adjustment"
                                    )) {
  base_model <- match.arg(base_model)
  variant <- match.arg(variant)
  
  hybrid_train <- build_direct_hybrid_base(
    base_results = train_base_results,
    xgb_results = xgb_train_results,
    data = data,
    h = h,
    base_model = base_model
  )
  
  hybrid_test <- build_direct_hybrid_base(
    base_results = test_base_results,
    xgb_results = xgb_test_results,
    data = data,
    h = h,
    base_model = base_model
  ) %>%
    arrange(forecast_origin)
  
  if (variant == "fixed_6040") {
    hybrid_test_final <- hybrid_test %>%
      rowwise() %>%
      mutate(
        weight_base = ifelse(h == 1, fixed_weight_base_h1, fixed_weight_base_other),
        weight_ml = 1 - weight_base,
        forecast_hybrid = weight_base * base_forecast + weight_ml * ml_forecast
      ) %>%
      ungroup()
    fit_obj <- list(weight_base = ifelse(h == 1, fixed_weight_base_h1, fixed_weight_base_other))
  } else if (variant == "rolling_rmse") {
    hybrid_test_final <- combine_rolling_rmse_fixed(hybrid_train, hybrid_test)
    fit_obj <- list(weight_base = estimate_static_weight_rmse(hybrid_train))
  } else if (variant == "linear_meta") {
    meta_fit <- fit_linear_meta_fixed(hybrid_train)
    hybrid_test_final <- combine_linear_meta_fixed(meta_fit, hybrid_test)
    fit_obj <- meta_fit
  } else if (variant == "regime_threshold") {
    hybrid_test_final <- hybrid_test %>%
      rowwise() %>%
      mutate(
        weight_base = ifelse(cpi_mm >= regime_inflation_threshold,
                             regime_weight_base_stress,
                             regime_weight_base_normal),
        weight_ml = 1 - weight_base,
        forecast_hybrid = weight_base * base_forecast + weight_ml * ml_forecast
      ) %>%
      ungroup()
    fit_obj <- list(
      threshold = regime_inflation_threshold,
      weight_base_normal = regime_weight_base_normal,
      weight_base_stress = regime_weight_base_stress
    )
  } else if (variant == "anchor_adjustment") {
    hybrid_test_final <- combine_anchor_adjustment_fixed(hybrid_train, hybrid_test)
    fit_obj <- list(gamma = estimate_anchor_gamma_fixed(hybrid_train))
  } else {
    stop("Unknown direct hybrid variant")
  }
  
  metrics <- compute_direct_hybrid_metrics(
    df = hybrid_test_final,
    base_model = base_model,
    variant_name = variant,
    h = h
  )
  
  list(
    results = hybrid_test_final,
    metrics = metrics,
    fit = fit_obj,
    hybrid_train_base = hybrid_train,
    hybrid_test_base = hybrid_test
  )
}

run_all_direct_hybrid_fixed <- function(train_base_results,
                                        test_base_results,
                                        xgb_train_results,
                                        xgb_test_results,
                                        data,
                                        h,
                                        base_model = c("TVP", "AR3", "OLS")) {
  base_model <- match.arg(base_model)
  
  variant_runs <- lapply(direct_hybrid_variants, function(v) {
    message("Direct hybrid fixed | base=", base_model, " | h=", h, " | variant=", v)
    run_direct_hybrid_fixed(
      train_base_results = train_base_results,
      test_base_results = test_base_results,
      xgb_train_results = xgb_train_results,
      xgb_test_results = xgb_test_results,
      data = data,
      h = h,
      base_model = base_model,
      variant = v
    )
  })
  names(variant_runs) <- direct_hybrid_variants
  
  metrics_all <- bind_rows(lapply(variant_runs, `[[`, "metrics"))
  
  list(
    results_by_variant = lapply(variant_runs, `[[`, "results"),
    fits_by_variant = lapply(variant_runs, `[[`, "fit"),
    metrics_all = metrics_all,
    hybrid_train_base = variant_runs[[1]]$hybrid_train_base,
    hybrid_test_base = variant_runs[[1]]$hybrid_test_base
  )
}

# =========================================================
# 7. BASE PIPELINE FOR ONE HORIZON
# =========================================================
run_fixed_base_models_one_h <- function(data, h,
                                        tvp_predictors = tvp_predictors,
                                        ols_formula = ols_formula,
                                        ar3_formula = ar3_formula,
                                        niter_tvp = niter_tvp,
                                        nburn_tvp = nburn_tvp) {
  df_h <- make_horizon_target(data, h) %>%
    arrange(date)
  
  train_df <- df_h %>% filter(date >= train_start, date <= train_end)
  test_df  <- df_h %>% filter(date >= test_start)
  
  # TVP
  tvp_fit <- fit_tvp_direct(train_df, predictors = tvp_predictors, niter = niter_tvp, nburn = nburn_tvp)
  tvp_train <- predict_tvp_fixed(tvp_fit, train_df, predictors = tvp_predictors)
  tvp_test  <- predict_tvp_fixed(tvp_fit, test_df, predictors = tvp_predictors)
  
  # OLS
  ols_fit <- fit_lm_direct(train_df, ols_formula)
  ols_train <- predict_lm_fixed(ols_fit, train_df, "forecast_ols")
  ols_test  <- predict_lm_fixed(ols_fit, test_df, "forecast_ols")
  
  # AR3
  ar3_fit <- fit_lm_direct(train_df, ar3_formula)
  ar3_train <- predict_lm_fixed(ar3_fit, train_df, "forecast_ar3")
  ar3_test  <- predict_lm_fixed(ar3_fit, test_df, "forecast_ar3")
  
  # ARIMA
  arima_fit <- fit_arima_direct(train_df)
  arima_train <- train_df %>%
    select(date, y_h) %>%
    rename(forecast_origin = date, actual = y_h) %>%
    mutate(forecast_arima = NA_real_)
  arima_test <- predict_arima_fixed(arima_fit, test_df)
  
  train_results <- train_df %>%
    select(date, y_h) %>%
    rename(forecast_origin = date, actual = y_h) %>%
    left_join(tvp_train$results %>% select(forecast_origin, forecast_tvp), by = "forecast_origin") %>%
    left_join(ols_train %>% select(forecast_origin, forecast_ols), by = "forecast_origin") %>%
    left_join(ar3_train %>% select(forecast_origin, forecast_ar3), by = "forecast_origin") %>%
    left_join(arima_train %>% select(forecast_origin, forecast_arima), by = "forecast_origin") %>%
    mutate(horizon = h) %>%
    arrange(forecast_origin)
  
  test_results <- test_df %>%
    select(date, y_h) %>%
    rename(forecast_origin = date, actual = y_h) %>%
    left_join(tvp_test$results %>% select(forecast_origin, forecast_tvp), by = "forecast_origin") %>%
    left_join(ols_test %>% select(forecast_origin, forecast_ols), by = "forecast_origin") %>%
    left_join(ar3_test %>% select(forecast_origin, forecast_ar3), by = "forecast_origin") %>%
    left_join(arima_test %>% select(forecast_origin, forecast_arima), by = "forecast_origin") %>%
    mutate(horizon = h) %>%
    arrange(forecast_origin)
  
  train_metrics <- compute_backtest_metrics(train_results) %>% mutate(horizon = h, sample = "train")
  test_metrics  <- compute_backtest_metrics(test_results)  %>% mutate(horizon = h, sample = "test")
  
  list(
    train_results = train_results,
    test_results = test_results,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    fits = list(
      tvp = tvp_fit,
      ols = ols_fit,
      ar3 = ar3_fit,
      arima = arima_fit
    ),
    tvp_summary_train = tvp_train$summary,
    tvp_summary_test = tvp_test$summary,
    beta_last_train = tvp_train$beta_last,
    beta_last_test = tvp_test$beta_last
  )
}

run_fixed_xgb_one_h <- function(data, predictors, h,
                                model_name = "XGB_DIRECT_BASE",
                                min_obs = min_xgb_obs,
                                max_depth = xgb_max_depth_default,
                                nrounds = xgb_nrounds,
                                eta = xgb_eta,
                                subsample = xgb_subsample,
                                colsample_bytree = xgb_colsample_bytree,
                                min_child_weight = 1,
                                gamma = 0,
                                lambda = 1,
                                alpha = 0,
                                max_delta_step = 0) {
  df_h <- make_horizon_target(data, h) %>%
    arrange(date)
  
  train_df <- df_h %>% filter(date >= train_start, date <= train_end)
  test_df  <- df_h %>% filter(date >= test_start)
  
  fit_obj <- fit_xgb_direct_fixed(
    train_df = train_df,
    predictors = predictors,
    min_obs = min_obs,
    max_depth = max_depth,
    nrounds = nrounds,
    eta = eta,
    subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha,
    max_delta_step = max_delta_step
  )
  
  train_results <- predict_xgb_fixed(fit_obj$fit, train_df, predictors, "forecast_xgb") %>%
    mutate(horizon = h, xgb_train_n = fit_obj$train_n) %>%
    arrange(forecast_origin)
  
  test_results <- predict_xgb_fixed(fit_obj$fit, test_df, predictors, "forecast_xgb") %>%
    mutate(horizon = h, xgb_train_n = fit_obj$train_n) %>%
    arrange(forecast_origin)
  
  train_metrics <- compute_xgb_direct_metrics(train_results, model_name = paste0(model_name, "_TRAIN"), h = h)
  train_metrics$sample <- "train"
  
  test_metrics <- compute_xgb_direct_metrics(test_results, model_name = model_name, h = h)
  test_metrics$sample <- "test"
  
  list(
    train_results = train_results,
    test_results = test_results,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    fit = fit_obj$fit,
    train_n = fit_obj$train_n,
    train_cc = fit_obj$train_cc
  )
}
run_residual_hybrid_shap_one_h <- function(h, residual_runs_h,
                                           base_models = hybrid_base_models,
                                           variants = hybrid_variants,
                                           predictors = ml_predictors) {
  shap_all <- list()
  global_all <- list()
  local_all <- list()
  
  for (bm in base_models) {
    runs_h_bm <- residual_runs_h[[bm]]$runs
    if (is.null(runs_h_bm)) next
    
    for (v in variants) {
      run_obj <- runs_h_bm[[v]]
      if (is.null(run_obj) || is.null(run_obj$fit)) next
      
      fit_obj <- run_obj$fit
      test_df <- run_obj$results
      
      if (is.null(test_df) || nrow(test_df) == 0) next
      
      shap_list <- lapply(seq_len(nrow(test_df)), function(i) {
        row_i <- test_df[i, , drop = FALSE]
        if (any(is.na(row_i[, predictors, drop = FALSE]))) return(NULL)
        
        x_mat <- as.matrix(row_i[, predictors, drop = FALSE])
        dmat <- xgb.DMatrix(data = x_mat)
        shap_vals <- tryCatch(
          predict(fit_obj, dmat, predcontrib = TRUE),
          error = function(e) NULL
        )
        if (is.null(shap_vals)) return(NULL)
        
        shap_df <- as_tibble(shap_vals)
        colnames(shap_df) <- c(predictors, "BIAS")
        
        shap_df %>%
          mutate(
            forecast_origin = as.Date(row_i$forecast_origin),
            horizon = h,
            base_model = bm,
            variant = v,
            model_group = "residual_hybrid",
            component = "residual_pred_ml"
          ) %>%
          pivot_longer(
            cols = all_of(c(predictors, "BIAS")),
            names_to = "feature",
            values_to = "shap_value"
          )
      })
      
      shap_df_variant <- bind_rows(shap_list)
      if (nrow(shap_df_variant) == 0) next
      
      global_df_variant <- shap_df_variant %>%
        filter(feature != "BIAS") %>%
        group_by(horizon, base_model, variant, model_group, component, feature) %>%
        summarise(
          mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
          mean_shap = mean(shap_value, na.rm = TRUE),
          sd_shap = sd(shap_value, na.rm = TRUE),
          n = n(),
          .groups = "drop"
        ) %>%
        arrange(horizon, base_model, variant, desc(mean_abs_shap))
      
      local_df_variant <- shap_df_variant %>%
        filter(feature != "BIAS") %>%
        group_by(base_model, variant, forecast_origin) %>%
        mutate(rank_abs_shap = min_rank(desc(abs(shap_value)))) %>%
        ungroup() %>%
        filter(rank_abs_shap <= 5)
      
      key <- paste(bm, v, sep = "__")
      shap_all[[key]] <- shap_df_variant
      global_all[[key]] <- global_df_variant
      local_all[[key]] <- local_df_variant
    }
  }
  
  list(
    shap_all = bind_rows(shap_all),
    global = bind_rows(global_all),
    local = bind_rows(local_all)
  )
}

# =========================================================
# 8. XAI HELPERS
# =========================================================
build_iml_predictor_xgb <- function(fit, data_df, predictors, y_col) {
  if (is.null(fit) || is.null(data_df) || nrow(data_df) == 0) return(NULL)
  
  x_df <- data_df %>%
    select(all_of(predictors)) %>%
    mutate(across(everything(), as.numeric))
  
  y_vec <- as.numeric(data_df[[y_col]])
  keep <- stats::complete.cases(x_df) & is.finite(y_vec)
  if (!any(keep)) return(NULL)
  
  x_df <- x_df[keep, , drop = FALSE]
  y_vec <- y_vec[keep]
  
  Predictor$new(
    model = fit,
    data = x_df,
    y = y_vec,
    predict.function = function(model, newdata) {
      newdata <- as.data.frame(newdata)
      x_mat <- as.matrix(newdata)
      as.numeric(predict(model, xgb.DMatrix(data = x_mat)))
    }
  )
}

ensure_xai_columns <- function(df, kind = c("ale", "interaction")) {
  kind <- match.arg(kind)
  
  if (is.null(df) || nrow(df) == 0) {
    if (kind == "ale") {
      return(tibble(
        horizon = integer(),
        model_group = character(),
        model_label = character(),
        base_model = character(),
        variant = character(),
        component = character(),
        feature = character(),
        feature_value = numeric(),
        ale = numeric()
      ))
    } else {
      return(tibble(
        horizon = integer(),
        model_group = character(),
        model_label = character(),
        base_model = character(),
        variant = character(),
        component = character(),
        feature = character(),
        interaction_strength = numeric()
      ))
    }
  }
  
  df <- as_tibble(df)
  
  if (!("horizon" %in% names(df))) df$horizon <- NA_integer_
  if (!("model_group" %in% names(df))) df$model_group <- NA_character_
  if (!("model_label" %in% names(df))) df$model_label <- NA_character_
  if (!("base_model" %in% names(df))) df$base_model <- NA_character_
  if (!("variant" %in% names(df))) df$variant <- NA_character_
  if (!("component" %in% names(df))) df$component <- NA_character_
  if (!("feature" %in% names(df))) df$feature <- NA_character_
  
  if (kind == "ale") {
    if (!("feature_value" %in% names(df))) df$feature_value <- NA_real_
    if (!("ale" %in% names(df))) df$ale <- NA_real_
    
    return(
      df %>%
        select(
          horizon, model_group, model_label, base_model, variant, component,
          feature, feature_value, ale
        )
    )
  } else {
    if (!("interaction_strength" %in% names(df))) df$interaction_strength <- NA_real_
    
    return(
      df %>%
        select(
          horizon, model_group, model_label, base_model, variant, component,
          feature, interaction_strength
        )
    )
  }
}

compute_xgb_ale_profiles <- function(fit, data_df, predictors, y_col,
                                     h, model_label, model_group,
                                     base_model = NA_character_,
                                     variant = NA_character_,
                                     grid_size = ale_grid_size) {
  predictor_obj <- build_iml_predictor_xgb(fit, data_df, predictors, y_col)
  if (is.null(predictor_obj)) return(tibble())
  
  ale_list <- lapply(predictors, function(pr) {
    eff <- tryCatch(
      FeatureEffect$new(
        predictor = predictor_obj,
        feature = pr,
        method = "ale",
        grid.size = grid_size
      ),
      error = function(e) NULL
    )
    
    if (is.null(eff)) return(NULL)
    dat <- tryCatch(as_tibble(eff$results), error = function(e) NULL)
    if (is.null(dat) || nrow(dat) == 0) return(NULL)
    if (!(pr %in% names(dat))) return(NULL)
    if (!(".value" %in% names(dat))) return(NULL)
    
    dat %>%
      transmute(
        horizon = h,
        model_group = model_group,
        model_label = model_label,
        base_model = base_model,
        variant = variant,
        feature = pr,
        feature_value = .data[[pr]],
        ale = .data$.value
      )
  })
  
  bind_rows(ale_list)
}

compute_xgb_feature_interactions <- function(fit, data_df, predictors, y_col,
                                             h, model_label, model_group,
                                             base_model = NA_character_,
                                             variant = NA_character_,
                                             top_n = interaction_top_n,
                                             sample_size = interaction_sample_size,
                                             seed = interaction_seed) {
  if (is.null(fit) || is.null(data_df) || nrow(data_df) == 0) return(tibble())
  
  data_use <- data_df %>%
    select(all_of(c(predictors, y_col))) %>%
    drop_na()
  
  if (nrow(data_use) == 0) return(tibble())
  
  if (!is.null(sample_size) && is.finite(sample_size) && nrow(data_use) > sample_size) {
    set.seed(seed)
    idx <- sort(sample(seq_len(nrow(data_use)), sample_size))
    data_use <- data_use[idx, , drop = FALSE]
  }
  
  predictor_obj <- build_iml_predictor_xgb(
    fit = fit,
    data_df = data_use,
    predictors = predictors,
    y_col = y_col
  )
  if (is.null(predictor_obj)) return(tibble())
  
  inter_obj <- tryCatch(
    Interaction$new(predictor_obj),
    error = function(e) NULL
  )
  if (is.null(inter_obj)) return(tibble())
  
  inter_df <- tryCatch(as_tibble(inter_obj$results), error = function(e) NULL)
  if (is.null(inter_df) || nrow(inter_df) == 0) return(tibble())
  
  feature_col <- if ("feature" %in% names(inter_df)) "feature" else names(inter_df)[1]
  value_col <- if (".interaction" %in% names(inter_df)) ".interaction" else names(inter_df)[2]
  
  inter_df %>%
    transmute(
      horizon = h,
      model_group = model_group,
      model_label = model_label,
      base_model = base_model,
      variant = variant,
      feature = .data[[feature_col]],
      interaction_strength = .data[[value_col]]
    ) %>%
    arrange(desc(interaction_strength)) %>%
    slice_head(n = top_n)
}

run_fixed_xgb_xai_one_h <- function(data, predictors, h, fit_obj,
                                    model_label = "XGB_DIRECT_BASE",
                                    model_group = "direct") {
  df_h <- make_horizon_target(data, h) %>% arrange(date)
  train_df <- df_h %>% filter(date >= train_start, date <= train_end)
  
  ale_df <- compute_xgb_ale_profiles(
    fit = fit_obj,
    data_df = train_df,
    predictors = predictors,
    y_col = "y_h",
    h = h,
    model_label = model_label,
    model_group = model_group
  )
  
  interaction_df <- compute_xgb_feature_interactions(
    fit = fit_obj,
    data_df = train_df,
    predictors = predictors,
    y_col = "y_h",
    h = h,
    model_label = model_label,
    model_group = model_group
  )
  
  list(
    ale = ale_df,
    interaction = interaction_df
  )
}

run_residual_hybrid_ale_interaction_one_h <- function(h, residual_runs_h,
                                                      base_models = hybrid_base_models,
                                                      variants = hybrid_variants,
                                                      predictors = ml_predictors) {
  ale_all <- list()
  interaction_all <- list()
  
  for (bm in base_models) {
    runs_h_bm <- residual_runs_h[[bm]]$runs
    if (is.null(runs_h_bm)) next
    
    for (v in variants) {
      run_obj <- runs_h_bm[[v]]
      if (is.null(run_obj)) next
      
      train_df <- run_obj$train_cc
      fit_obj <- run_obj$fit
      if (is.null(train_df) || nrow(train_df) == 0 || is.null(fit_obj)) next
      
      model_label <- paste0("RESIDUAL_", bm, "_", v)
      
      ale_all[[paste(bm, v, sep = "__")]] <- compute_xgb_ale_profiles(
        fit = fit_obj,
        data_df = train_df,
        predictors = predictors,
        y_col = "residual",
        h = h,
        model_label = model_label,
        model_group = "residual_hybrid",
        base_model = bm,
        variant = v
      ) %>%
        mutate(component = "residual_pred_ml")
      
      interaction_all[[paste(bm, v, sep = "__")]] <- compute_xgb_feature_interactions(
        fit = fit_obj,
        data_df = train_df,
        predictors = predictors,
        y_col = "residual",
        h = h,
        model_label = model_label,
        model_group = "residual_hybrid",
        base_model = bm,
        variant = v
      ) %>%
        mutate(component = "residual_pred_ml")
    }
  }
  
  list(
    ale = bind_rows(ale_all),
    interaction = bind_rows(interaction_all)
  )
}

compute_one_xgb_shap <- function(fit, newdata_row, predictors, origin_date) {
  if (is.null(fit)) return(NULL)
  if (nrow(newdata_row) == 0) return(NULL)
  if (any(is.na(newdata_row[, predictors, drop = FALSE]))) return(NULL)
  
  x_mat <- as.matrix(newdata_row[, predictors, drop = FALSE])
  dmat <- xgb.DMatrix(data = x_mat)
  shap_vals <- predict(fit, dmat, predcontrib = TRUE)
  
  shap_df <- as_tibble(shap_vals)
  colnames(shap_df) <- c(predictors, "BIAS")
  
  shap_df %>%
    mutate(forecast_origin = as.Date(origin_date)) %>%
    pivot_longer(
      cols = all_of(c(predictors, "BIAS")),
      names_to = "feature",
      values_to = "shap_value"
    )
}

run_fixed_xgb_shap_one_h <- function(data, predictors, h, fit_obj) {
  df_h <- make_horizon_target(data, h) %>% arrange(date)
  test_df <- df_h %>% filter(date >= test_start)
  
  shap_list <- lapply(seq_len(nrow(test_df)), function(i) {
    compute_one_xgb_shap(
      fit = fit_obj,
      newdata_row = test_df[i, , drop = FALSE],
      predictors = predictors,
      origin_date = test_df$date[i]
    )
  })
  
  shap_df <- bind_rows(shap_list) %>% mutate(horizon = h)
  
  global_df <- shap_df %>%
    filter(feature != "BIAS") %>%
    group_by(horizon, feature) %>%
    summarise(
      mean_abs_shap = mean(abs(shap_value), na.rm = TRUE),
      mean_shap = mean(shap_value, na.rm = TRUE),
      sd_shap = sd(shap_value, na.rm = TRUE),
      n = n(),
      .groups = "drop"
    ) %>%
    arrange(horizon, desc(mean_abs_shap))
  
  local_df <- shap_df %>%
    filter(feature != "BIAS") %>%
    group_by(forecast_origin) %>%
    mutate(rank_abs_shap = min_rank(desc(abs(shap_value)))) %>%
    ungroup() %>%
    filter(rank_abs_shap <= 5)
  
  list(
    shap_all = shap_df,
    global = global_df,
    local = local_df
  )
}

compare_tvp_vs_ml_all_test_dates <- function(tvp_beta_last_df, shap_df, h, shap_threshold = 0.05) {
  if (nrow(tvp_beta_last_df) == 0 || nrow(shap_df) == 0) return(tibble())
  
  shap_dates <- unique(as.Date(shap_df$forecast_origin))
  
  bind_rows(lapply(shap_dates, function(d) {
    tvp_df <- tvp_beta_last_df %>%
      filter(forecast_origin == d) %>%
      select(predictor, beta_last_mean, beta_last_nonzero_95)
    
    shap_date_df <- shap_df %>%
      filter(forecast_origin == d, feature != "BIAS") %>%
      rename(predictor = feature) %>%
      select(predictor, shap_value)
    
    tvp_df %>%
      inner_join(shap_date_df, by = "predictor") %>%
      mutate(
        horizon = h,
        tvp_sign = sign(beta_last_mean),
        shap_sign = sign(shap_value),
        consistent_sign = (tvp_sign == shap_sign),
        importance_ml = abs(shap_value),
        importance_tvp = abs(beta_last_mean),
        classification = case_when(
          beta_last_nonzero_95 & consistent_sign ~ "consistent",
          !beta_last_nonzero_95 & abs(shap_value) > shap_threshold ~ "ml_only",
          beta_last_nonzero_95 & abs(shap_value) < 0.01 ~ "tvp_only",
          beta_last_nonzero_95 & !consistent_sign ~ "conflicting",
          TRUE ~ "weak"
        )
      ) %>%
      arrange(desc(abs(shap_value)))
  }))
}

run_direct_hybrid_xai_one_h <- function(h, direct_runs_h, shap_df_h,
                                        base_models = c("TVP", "AR3", "OLS"),
                                        selected_variants = c("fixed_6040", "rolling_rmse", "linear_meta", "regime_threshold")) {
  add_effective_weights <- function(df, variant_name) {
    df <- df %>% mutate(denom = base_forecast - ml_forecast)
    
    if (variant_name %in% c("fixed_6040", "rolling_rmse", "regime_threshold")) {
      df <- df %>%
        mutate(
          effective_weight_base = weight_base,
          effective_weight_ml = weight_ml
        )
    } else if (variant_name == "linear_meta") {
      df <- df %>%
        mutate(
          effective_weight_base = ifelse(
            is.na(denom) | abs(denom) < 1e-8,
            NA_real_,
            (forecast_hybrid - ml_forecast) / denom
          ),
          effective_weight_ml = ifelse(is.na(effective_weight_base), NA_real_, 1 - effective_weight_base)
        )
    } else {
      df <- df %>%
        mutate(
          effective_weight_base = NA_real_,
          effective_weight_ml = NA_real_
        )
    }
    
    df %>%
      mutate(
        contribution_base = effective_weight_base * base_forecast,
        contribution_ml = effective_weight_ml * ml_forecast,
        dominant_layer = case_when(
          is.na(effective_weight_base) | is.na(effective_weight_ml) ~ "unknown",
          effective_weight_base >= 0.51 ~ "base",
          effective_weight_ml >= 0.51 ~ "ml",
          TRUE ~ "mixed"
        ),
        variant = variant_name
      )
  }
  
  global_list <- list()
  local_list <- list()
  ml_dominant_shap_list <- list()
  
  for (bm in base_models) {
    runs_h_bm <- direct_runs_h[[bm]]$results_by_variant
    
    xai_df <- bind_rows(
      lapply(selected_variants, function(v) {
        add_effective_weights(runs_h_bm[[v]], v)
      })
    ) %>%
      mutate(horizon = h, base_model = bm) %>%
      arrange(variant, forecast_origin)
    
    global_df <- xai_df %>%
      group_by(horizon, base_model, variant) %>%
      summarise(
        n = n(),
        mean_weight_base = mean(effective_weight_base, na.rm = TRUE),
        mean_weight_ml = mean(effective_weight_ml, na.rm = TRUE),
        share_base = mean(dominant_layer == "base", na.rm = TRUE),
        share_ml = mean(dominant_layer == "ml", na.rm = TRUE),
        share_mixed = mean(dominant_layer == "mixed", na.rm = TRUE),
        mean_contribution_base = mean(contribution_base, na.rm = TRUE),
        mean_contribution_ml = mean(contribution_ml, na.rm = TRUE),
        .groups = "drop"
      )
    
    local_df <- xai_df %>%
      group_by(variant, forecast_origin) %>%
      slice_head(n = 1) %>%
      ungroup()
    
    ml_dom <- xai_df %>%
      filter(dominant_layer == "ml") %>%
      select(horizon, base_model, variant, forecast_origin, base_forecast, ml_forecast, forecast_hybrid)
    shap_wide_h <- shap_df_h %>%
      filter(feature != "BIAS") %>%
      select(forecast_origin, feature, shap_value) %>%
      distinct() %>%
      pivot_wider(
        names_from = feature,
        values_from = shap_value,
        names_prefix = "shap_"
      )
    
    if (nrow(ml_dom) > 0 && nrow(shap_wide_h) > 0) {
      ml_dom_shap <- ml_dom %>%
        left_join(
          shap_wide_h,
          by = "forecast_origin"
        )
    } else {
      ml_dom_shap <- tibble()
    }
    
    global_list[[bm]] <- global_df
    local_list[[bm]] <- local_df
    ml_dominant_shap_list[[bm]] <- ml_dom_shap
  }
  
  list(
    global = bind_rows(global_list),
    local = bind_rows(local_list),
    ml_dominant_with_shap = bind_rows(ml_dominant_shap_list)
  )
}

run_residual_hybrid_xai_one_h <- function(h, residual_runs_h, residual_shap_h,
                                          base_models = c("TVP", "AR3", "OLS"),
                                          selected_variants = c("last10y", "weighted", "shallow", "unrestricted")) {
  all_global <- list()
  all_local <- list()
  all_shap_link <- list()
  all_full <- list()
  
  for (bm in base_models) {
    runs_h_bm <- residual_runs_h[[bm]]$runs
    
    full_df <- bind_rows(
      lapply(selected_variants, function(v) {
        df <- runs_h_bm[[v]]$results
        
        df %>%
          mutate(
            horizon = h,
            base_model = bm,
            variant = v,
            correction_ml = residual_pred_ml,
            correction_abs = abs(residual_pred_ml),
            correction_direction = case_when(
              is.na(residual_pred_ml) ~ "unknown",
              residual_pred_ml > 0 ~ "upward",
              residual_pred_ml < 0 ~ "downward",
              TRUE ~ "zero"
            ),
            error_base = base_forecast - actual,
            error_hybrid = forecast_hybrid - actual,
            correction_improves_abs_error = abs(error_hybrid) < abs(error_base),
            correction_worsens_abs_error = abs(error_hybrid) > abs(error_base),
            ml_to_base_ratio = ifelse(
              is.na(base_forecast) | abs(base_forecast) < 1e-8,
              NA_real_,
              residual_pred_ml / base_forecast
            )
          )
      })
    ) %>% arrange(variant, forecast_origin)
    
    global_df <- full_df %>%
      group_by(horizon, base_model, variant) %>%
      summarise(
        n = n(),
        mean_base_forecast = mean(base_forecast, na.rm = TRUE),
        mean_hybrid_forecast = mean(forecast_hybrid, na.rm = TRUE),
        mean_correction_ml = mean(correction_ml, na.rm = TRUE),
        mean_abs_correction_ml = mean(correction_abs, na.rm = TRUE),
        median_abs_correction_ml = median(correction_abs, na.rm = TRUE),
        sd_correction_ml = sd(correction_ml, na.rm = TRUE),
        share_upward = mean(correction_direction == "upward", na.rm = TRUE),
        share_downward = mean(correction_direction == "downward", na.rm = TRUE),
        share_zero = mean(correction_direction == "zero", na.rm = TRUE),
        share_improves_abs_error = mean(correction_improves_abs_error, na.rm = TRUE),
        share_worsens_abs_error = mean(correction_worsens_abs_error, na.rm = TRUE),
        .groups = "drop"
      )
    
    local_df <- full_df %>%
      group_by(variant, forecast_origin) %>%
      slice_head(n = 1) %>%
      ungroup()
    
    shap_wide_h <- residual_shap_h %>%
      filter(base_model == bm, feature != "BIAS") %>%
      select(base_model, variant, forecast_origin, feature, shap_value) %>%
      distinct() %>%
      pivot_wider(
        names_from = feature,
        values_from = shap_value,
        names_prefix = "shap_"
      )
    
    shap_link <- full_df %>%
      select(horizon, base_model, variant, forecast_origin, correction_ml, forecast_hybrid) %>%
      left_join(
        shap_wide_h,
        by = c("base_model", "variant", "forecast_origin")
      )
    
    all_full[[bm]] <- full_df
    all_global[[bm]] <- global_df
    all_local[[bm]] <- local_df
    all_shap_link[[bm]] <- shap_link
  }
  
  list(
    full = bind_rows(all_full),
    global = bind_rows(all_global),
    local = bind_rows(all_local),
    shap_link = bind_rows(all_shap_link)
  )
}

# =========================================================
# 8A. DIRECT HYBRID XAI FOR HYBRID FORECAST ITSELF
# =========================================================

build_iml_predictor_custom <- function(data_df, predictors, y_col, predict_fun) {
  if (is.null(data_df) || nrow(data_df) == 0) return(NULL)
  
  x_df <- data_df %>%
    select(all_of(predictors)) %>%
    mutate(across(everything(), as.numeric))
  
  y_vec <- as.numeric(data_df[[y_col]])
  keep <- stats::complete.cases(x_df) & is.finite(y_vec)
  if (!any(keep)) return(NULL)
  
  x_df <- x_df[keep, , drop = FALSE]
  y_vec <- y_vec[keep]
  
  Predictor$new(
    model = list(dummy = TRUE),
    data = x_df,
    y = y_vec,
    predict.function = function(model, newdata) {
      predict_fun(as.data.frame(newdata))
    }
  )
}

compute_custom_ale_profiles <- function(data_df, predictors, y_col, predict_fun,
                                        h, model_label, model_group,
                                        base_model = NA_character_,
                                        variant = NA_character_,
                                        component = NA_character_,
                                        grid_size = ale_grid_size) {
  predictor_obj <- build_iml_predictor_custom(
    data_df = data_df,
    predictors = predictors,
    y_col = y_col,
    predict_fun = predict_fun
  )
  if (is.null(predictor_obj)) return(tibble())
  
  ale_list <- lapply(predictors, function(pr) {
    eff <- tryCatch(
      FeatureEffect$new(
        predictor = predictor_obj,
        feature = pr,
        method = "ale",
        grid.size = grid_size
      ),
      error = function(e) NULL
    )
    
    if (is.null(eff)) return(NULL)
    dat <- tryCatch(as_tibble(eff$results), error = function(e) NULL)
    if (is.null(dat) || nrow(dat) == 0) return(NULL)
    if (!(pr %in% names(dat))) return(NULL)
    if (!(".value" %in% names(dat))) return(NULL)
    
    dat %>%
      transmute(
        horizon = h,
        model_group = model_group,
        model_label = model_label,
        base_model = base_model,
        variant = variant,
        component = component,
        feature = pr,
        feature_value = .data[[pr]],
        ale = .data$.value
      )
  })
  
  bind_rows(ale_list)
}

compute_custom_feature_interactions <- function(data_df, predictors, y_col, predict_fun,
                                                h, model_label, model_group,
                                                base_model = NA_character_,
                                                variant = NA_character_,
                                                component = NA_character_,
                                                top_n = interaction_top_n,
                                                sample_size = interaction_sample_size,
                                                seed = interaction_seed) {
  if (is.null(data_df) || nrow(data_df) == 0) return(tibble())
  
  data_use <- data_df %>%
    select(all_of(c(predictors, y_col))) %>%
    drop_na()
  
  if (nrow(data_use) == 0) return(tibble())
  
  if (!is.null(sample_size) && is.finite(sample_size) && nrow(data_use) > sample_size) {
    set.seed(seed)
    idx <- sort(sample(seq_len(nrow(data_use)), sample_size))
    data_use <- data_use[idx, , drop = FALSE]
  }
  
  predictor_obj <- build_iml_predictor_custom(
    data_df = data_use,
    predictors = predictors,
    y_col = y_col,
    predict_fun = predict_fun
  )
  if (is.null(predictor_obj)) return(tibble())
  
  inter_obj <- tryCatch(
    Interaction$new(predictor_obj),
    error = function(e) NULL
  )
  if (is.null(inter_obj)) return(tibble())
  
  inter_df <- tryCatch(as_tibble(inter_obj$results), error = function(e) NULL)
  if (is.null(inter_df) || nrow(inter_df) == 0) return(tibble())
  
  feature_col <- if ("feature" %in% names(inter_df)) "feature" else names(inter_df)[1]
  value_col <- if (".interaction" %in% names(inter_df)) ".interaction" else names(inter_df)[2]
  
  inter_df %>%
    transmute(
      horizon = h,
      model_group = model_group,
      model_label = model_label,
      base_model = base_model,
      variant = variant,
      component = component,
      feature = .data[[feature_col]],
      interaction_strength = .data[[value_col]]
    ) %>%
    arrange(desc(interaction_strength)) %>%
    slice_head(n = top_n)
}

# =========================================================
# 8B. DIRECT HYBRID ALE / INTERACTION
# =========================================================

run_direct_hybrid_ale_interaction_one_h <- function(h, direct_runs_h,
                                                    base_models = hybrid_base_models,
                                                    variants = direct_hybrid_variants,
                                                    predictors = ml_predictors) {
  ale_all <- list()
  interaction_all <- list()
  
  for (bm in base_models) {
    runs_h_bm <- direct_runs_h[[bm]]$results_by_variant
    if (is.null(runs_h_bm)) next
    
    for (v in variants) {
      df_variant <- runs_h_bm[[v]]
      if (is.null(df_variant) || nrow(df_variant) == 0) next
      
      df_train_h <- make_horizon_target(data, h) %>%
        filter(date >= train_start, date <= train_end) %>%
        arrange(date)
      
      df_xai <- df_train_h %>%
        select(date, all_of(predictors), y_h, cpi_mm) %>%
        drop_na()
      
      if (nrow(df_xai) == 0) next
      
      model_label_ml <- paste0("DIRECT_", bm, "_", v, "_ML")
      model_label_hybrid <- paste0("DIRECT_", bm, "_", v, "_HYBRID")
      
      fit_xgb <- all_xgb_base_runs[[paste0("h", h)]]$fit
      xgb_train_cc <- all_xgb_base_runs[[paste0("h", h)]]$train_cc
      
      if (!is.null(fit_xgb) && !is.null(xgb_train_cc) && nrow(xgb_train_cc) > 0) {
        ale_ml <- compute_xgb_ale_profiles(
          fit = fit_xgb,
          data_df = xgb_train_cc,
          predictors = predictors,
          y_col = "y_h",
          h = h,
          model_label = model_label_ml,
          model_group = "direct_hybrid",
          base_model = bm,
          variant = v
        ) %>%
          mutate(component = "ml_forecast")
        
        inter_ml <- compute_xgb_feature_interactions(
          fit = fit_xgb,
          data_df = xgb_train_cc,
          predictors = predictors,
          y_col = "y_h",
          h = h,
          model_label = model_label_ml,
          model_group = "direct_hybrid",
          base_model = bm,
          variant = v
        ) %>%
          mutate(component = "ml_forecast")
      } else {
        ale_ml <- tibble()
        inter_ml <- tibble()
      }
      
      predict_fun_hybrid <- switch(
        v,
        "fixed_6040" = function(newdata) {
          w_base <- ifelse(h == 1, fixed_weight_base_h1, fixed_weight_base_other)
          base_col <- get_base_forecast_col(bm)
          
          base_fit_run <- all_base_runs[[paste0("h", h)]]$train_results %>%
            select(forecast_origin, all_of(base_col)) %>%
            rename(base_forecast = all_of(base_col))
          
          base_df <- make_horizon_target(data, h) %>%
            filter(date >= train_start, date <= train_end) %>%
            arrange(date)
          
          base_align <- base_df %>%
            transmute(forecast_origin = date) %>%
            left_join(base_fit_run, by = "forecast_origin")
          
          ml_pred <- as.numeric(predict(fit_xgb, xgb.DMatrix(as.matrix(newdata[, predictors, drop = FALSE]))))
          base_vec <- base_align$base_forecast[seq_len(nrow(newdata))]
          w_base * base_vec + (1 - w_base) * ml_pred
        },
        
        "rolling_rmse" = function(newdata) {
          w_base <- estimate_static_weight_rmse(
            train_history = direct_runs_h[[bm]]$hybrid_train_base,
            window_months = rolling_window_months,
            weight_lower = rolling_weight_lower,
            weight_upper = rolling_weight_upper
          )
          
          base_col <- get_base_forecast_col(bm)
          base_fit_run <- all_base_runs[[paste0("h", h)]]$test_results %>%
            select(forecast_origin, all_of(base_col)) %>%
            rename(base_forecast = all_of(base_col))
          
          base_df <- make_horizon_target(data, h) %>%
            filter(date >= test_start) %>%
            arrange(date)
          
          base_align <- base_df %>%
            transmute(forecast_origin = date) %>%
            left_join(base_fit_run, by = "forecast_origin")
          
          ml_pred <- as.numeric(predict(fit_xgb, xgb.DMatrix(as.matrix(newdata[, predictors, drop = FALSE]))))
          base_vec <- base_align$base_forecast[seq_len(nrow(newdata))]
          w_base * base_vec + (1 - w_base) * ml_pred
        },
        
        "linear_meta" = function(newdata) {
          meta_fit <- direct_runs_h[[bm]]$fits_by_variant[["linear_meta"]]
          base_col <- get_base_forecast_col(bm)
          
          base_fit_run <- all_base_runs[[paste0("h", h)]]$test_results %>%
            select(forecast_origin, all_of(base_col)) %>%
            rename(base_forecast = all_of(base_col))
          
          base_df <- make_horizon_target(data, h) %>%
            filter(date >= test_start) %>%
            arrange(date)
          
          base_align <- base_df %>%
            transmute(forecast_origin = date) %>%
            left_join(base_fit_run, by = "forecast_origin")
          
          ml_pred <- as.numeric(predict(fit_xgb, xgb.DMatrix(as.matrix(newdata[, predictors, drop = FALSE]))))
          base_vec <- base_align$base_forecast[seq_len(nrow(newdata))]
          
          tmp_df <- tibble(
            base_forecast = base_vec,
            ml_forecast = ml_pred
          )
          
          as.numeric(predict(meta_fit, newdata = tmp_df))
        },
        
        "regime_threshold" = function(newdata) {
          base_col <- get_base_forecast_col(bm)
          
          base_fit_run <- all_base_runs[[paste0("h", h)]]$test_results %>%
            select(forecast_origin, all_of(base_col)) %>%
            rename(base_forecast = all_of(base_col))
          
          base_df <- make_horizon_target(data, h) %>%
            filter(date >= test_start) %>%
            arrange(date)
          
          base_align <- base_df %>%
            transmute(forecast_origin = date, cpi_mm = cpi_mm) %>%
            left_join(base_fit_run, by = "forecast_origin")
          
          ml_pred <- as.numeric(predict(fit_xgb, xgb.DMatrix(as.matrix(newdata[, predictors, drop = FALSE]))))
          base_vec <- base_align$base_forecast[seq_len(nrow(newdata))]
          cpi_vec <- base_align$cpi_mm[seq_len(nrow(newdata))]
          
          w_base <- ifelse(
            cpi_vec >= regime_inflation_threshold,
            regime_weight_base_stress,
            regime_weight_base_normal
          )
          w_base * base_vec + (1 - w_base) * ml_pred
        },
        
        "anchor_adjustment" = function(newdata) {
          gamma <- estimate_anchor_gamma_fixed(
            train_history = direct_runs_h[[bm]]$hybrid_train_base,
            window_months = anchor_window_months,
            gamma_lower = anchor_gamma_lower,
            gamma_upper = anchor_gamma_upper
          )
          
          base_col <- get_base_forecast_col(bm)
          base_fit_run <- all_base_runs[[paste0("h", h)]]$test_results %>%
            select(forecast_origin, all_of(base_col)) %>%
            rename(base_forecast = all_of(base_col))
          
          base_df <- make_horizon_target(data, h) %>%
            filter(date >= test_start) %>%
            arrange(date)
          
          base_align <- base_df %>%
            transmute(forecast_origin = date) %>%
            left_join(base_fit_run, by = "forecast_origin")
          
          ml_pred <- as.numeric(predict(fit_xgb, xgb.DMatrix(as.matrix(newdata[, predictors, drop = FALSE]))))
          base_vec <- base_align$base_forecast[seq_len(nrow(newdata))]
          base_vec + gamma * (ml_pred - base_vec)
        },
        
        stop("Unknown direct hybrid variant in XAI")
      )
      
      ale_hybrid <- compute_custom_ale_profiles(
        data_df = df_xai,
        predictors = predictors,
        y_col = "y_h",
        predict_fun = predict_fun_hybrid,
        h = h,
        model_label = model_label_hybrid,
        model_group = "direct_hybrid",
        base_model = bm,
        variant = v,
        component = "forecast_hybrid"
      )
      
      inter_hybrid <- compute_custom_feature_interactions(
        data_df = df_xai,
        predictors = predictors,
        y_col = "y_h",
        predict_fun = predict_fun_hybrid,
        h = h,
        model_label = model_label_hybrid,
        model_group = "direct_hybrid",
        base_model = bm,
        variant = v,
        component = "forecast_hybrid"
      )
      
      ale_all[[paste(bm, v, "ml", sep = "__")]] <- ale_ml
      ale_all[[paste(bm, v, "hybrid", sep = "__")]] <- ale_hybrid
      interaction_all[[paste(bm, v, "ml", sep = "__")]] <- inter_ml
      interaction_all[[paste(bm, v, "hybrid", sep = "__")]] <- inter_hybrid
    }
  }
  
  list(
    ale = bind_rows(ale_all),
    interaction = bind_rows(interaction_all)
  )
}

# =========================================================
# 8C. SAVE XAI PLOTS
# =========================================================

save_ale_plots_pdf <- function(ale_df, subdir) {
  ale_df <- ensure_xai_columns(ale_df, kind = "ale")
  if (is.null(ale_df) || nrow(ale_df) == 0) return(invisible(NULL))
  
  out_dir <- file.path(xai_plot_dir, subdir, "ale")
  safe_dir_create(out_dir)
  
  split_keys <- ale_df %>%
    distinct(horizon, model_label, base_model, variant, component, feature)
  
  for (i in seq_len(nrow(split_keys))) {
    key <- split_keys[i, ]
    df_i <- ale_df %>%
      filter(
        horizon == key$horizon,
        model_label == key$model_label,
        ((is.na(base_model) & is.na(key$base_model)) | base_model == key$base_model),
        ((is.na(variant) & is.na(key$variant)) | variant == key$variant),
        ((is.na(component) & is.na(key$component)) | component == key$component),
        feature == key$feature
      )
    
    if (nrow(df_i) == 0) next
    
    p <- ggplot(df_i, aes(x = feature_value, y = ale)) +
      geom_line(linewidth = 0.8) +
      labs(
        title = paste0("ALE: ", key$model_label, " | h=", key$horizon, " | ", key$feature),
        x = key$feature,
        y = "ALE"
      ) +
      theme_minimal()
    
    file_name <- paste0(
      "ale_h", key$horizon, "_",
      sanitize_filename(dplyr::coalesce(key$model_label, "NA")), "_",
      sanitize_filename(dplyr::coalesce(key$base_model, "NA")), "_",
      sanitize_filename(dplyr::coalesce(key$variant, "NA")), "_",
      sanitize_filename(dplyr::coalesce(key$component, "NA")), "_",
      sanitize_filename(key$feature),
      ".pdf"
    )
    
    ggsave(
      filename = file.path(out_dir, file_name),
      plot = p,
      device = "pdf",
      width = 8,
      height = 5
    )
  }
  
  invisible(NULL)
}

save_interaction_plots_pdf <- function(interaction_df, subdir) {
  interaction_df <- ensure_xai_columns(interaction_df, kind = "interaction")
  if (is.null(interaction_df) || nrow(interaction_df) == 0) return(invisible(NULL))
  
  out_dir <- file.path(xai_plot_dir, subdir, "interaction")
  safe_dir_create(out_dir)
  
  split_keys <- interaction_df %>%
    distinct(horizon, model_label, base_model, variant, component)
  
  for (i in seq_len(nrow(split_keys))) {
    key <- split_keys[i, ]
    df_i <- interaction_df %>%
      filter(
        horizon == key$horizon,
        model_label == key$model_label,
        ((is.na(base_model) & is.na(key$base_model)) | base_model == key$base_model),
        ((is.na(variant) & is.na(key$variant)) | variant == key$variant),
        ((is.na(component) & is.na(key$component)) | component == key$component)
      ) %>%
      arrange(interaction_strength)
    
    if (nrow(df_i) == 0) next
    
    p <- ggplot(df_i, aes(x = interaction_strength, y = reorder(feature, interaction_strength))) +
      geom_col() +
      labs(
        title = paste0("Feature Interaction: ", key$model_label, " | h=", key$horizon),
        x = "Interaction strength",
        y = "Feature"
      ) +
      theme_minimal()
    
    file_name <- paste0(
      "interaction_h", key$horizon, "_",
      sanitize_filename(dplyr::coalesce(key$model_label, "NA")), "_",
      sanitize_filename(dplyr::coalesce(key$base_model, "NA")), "_",
      sanitize_filename(dplyr::coalesce(key$variant, "NA")), "_",
      sanitize_filename(dplyr::coalesce(key$component, "NA")),
      ".pdf"
    )
    
    ggsave(
      filename = file.path(out_dir, file_name),
      plot = p,
      device = "pdf",
      width = 8,
      height = 5
    )
  }
  
  invisible(NULL)
}

# =========================================================
# 9. FULL FIXED-MODEL PIPELINE
# =========================================================
all_base_runs <- list()
all_xgb_base_runs <- list()
all_xgb_rich_runs <- list()

all_residual_hybrid_runs <- list()
all_direct_hybrid_runs <- list()

all_tvp_global <- list()
all_xgb_shap <- list()
all_tvp_vs_ml <- list()
all_residual_hybrid_xai <- list()
all_direct_hybrid_xai <- list()
all_xgb_ale_interaction <- list()
all_xgb_rich_ale_interaction <- list()
all_residual_hybrid_ale_interaction <- list()
all_direct_hybrid_ale_interaction <- list()
all_ols_coefs <- list()
all_ar3_coefs <- list()

for (hh in horizons) {
  cat("\n====================================================\n")
  cat("RUN FIXED-MODEL PIPELINE FOR H =", hh, "\n")
  cat("====================================================\n")
  
  # 9.1 base econometric models
  base_run_h <- run_fixed_base_models_one_h(
    data = data,
    h = hh,
    tvp_predictors = tvp_predictors,
    ols_formula = ols_formula,
    ar3_formula = ar3_formula,
    niter_tvp = niter_tvp,
    nburn_tvp = nburn_tvp
  )
  all_base_runs[[paste0("h", hh)]] <- base_run_h
  
  # cat("\n--- COEFFICIENTS: horizon =", hh, "---\n")
  # 
  # cat("\nOLS coefficients:\n")
  # print(coef(base_run_h$fits$ols))
  # 
  # cat("\nAR3 coefficients:\n")
  # print(coef(base_run_h$fits$ar3))
  
  ols_coefs_h <- extract_lm_coef_table(base_run_h$fits$ols, hh, "OLS")
  ar3_coefs_h <- extract_lm_coef_table(base_run_h$fits$ar3, hh, "AR3")
  
  all_ols_coefs[[paste0("h", hh)]] <- ols_coefs_h
  all_ar3_coefs[[paste0("h", hh)]] <- ar3_coefs_h
  
  cat("\nOLS coefficients, h =", hh, "\n")
  print(ols_coefs_h)
  
  cat("\nAR3 coefficients, h =", hh, "\n")
  print(ar3_coefs_h)
  
  write_csv(base_run_h$train_results, paste0("fixed_base_train_results_h", hh, ".csv"))
  write_csv(base_run_h$test_results, paste0("fixed_base_test_results_h", hh, ".csv"))
  write_csv(base_run_h$train_metrics, paste0("fixed_base_train_metrics_h", hh, ".csv"))
  write_csv(base_run_h$test_metrics, paste0("fixed_base_test_metrics_h", hh, ".csv"))
  
  write_csv(base_run_h$tvp_summary_test, paste0("fixed_tvp_forecast_summary_test_h", hh, ".csv"))
  write_csv(base_run_h$beta_last_test, paste0("fixed_tvp_beta_last_test_h", hh, ".csv"))
  
  tvp_global_h <- compute_tvp_global_contribution(base_run_h$beta_last_test, hh)
  all_tvp_global[[paste0("h", hh)]] <- tvp_global_h
  write_csv(tvp_global_h, paste0("fixed_tvp_global_contribution_h", hh, ".csv"))
  
  # 9.2 XGB base
  xgb_params_h <- get_xgb_direct_params(hh)
  
  xgb_run_h <- run_fixed_xgb_one_h(
    data = data,
    predictors = ml_predictors,
    h = hh,
    model_name = "XGB_DIRECT_BASE",
    max_depth = xgb_params_h$max_depth,
    nrounds = xgb_params_h$nrounds,
    eta = xgb_params_h$eta,
    subsample = xgb_params_h$subsample,
    colsample_bytree = xgb_params_h$colsample_bytree,
    min_child_weight = xgb_params_h$min_child_weight,
    gamma = xgb_params_h$gamma,
    lambda = xgb_params_h$lambda,
    alpha = xgb_params_h$alpha,
    max_delta_step = xgb_params_h$max_delta_step
  )
  all_xgb_base_runs[[paste0("h", hh)]] <- xgb_run_h
  write_csv(xgb_run_h$train_results, paste0("fixed_xgb_direct_base_train_h", hh, ".csv"))
  write_csv(xgb_run_h$test_results, paste0("fixed_xgb_direct_base_test_h", hh, ".csv"))
  write_csv(xgb_run_h$train_metrics, paste0("fixed_xgb_direct_base_train_metrics_h", hh, ".csv"))
  write_csv(xgb_run_h$test_metrics, paste0("fixed_xgb_direct_base_test_metrics_h", hh, ".csv"))
  
  # 9.3 XGB rich
  # xgb_rich_run_h <- run_fixed_xgb_one_h(
  #   data = data_rich,
  #   predictors = rich_predictors,
  #   h = hh,
  #   model_name = "XGB_DIRECT_RICH"
  # )
  # all_xgb_rich_runs[[paste0("h", hh)]] <- xgb_rich_run_h
  # write_csv(xgb_rich_run_h$train_results, paste0("fixed_xgb_direct_rich_train_h", hh, ".csv"))
  # write_csv(xgb_rich_run_h$test_results, paste0("fixed_xgb_direct_rich_test_h", hh, ".csv"))
  # write_csv(xgb_rich_run_h$train_metrics, paste0("fixed_xgb_direct_rich_train_metrics_h", hh, ".csv"))
  # write_csv(xgb_rich_run_h$test_metrics, paste0("fixed_xgb_direct_rich_test_metrics_h", hh, ".csv"))
  
  # xgb_rich_xai_h <- run_fixed_xgb_xai_one_h(
  #   data = data_rich,
  #   predictors = rich_predictors,
  #   h = hh,
  #   fit_obj = xgb_rich_run_h$fit,
  #   model_label = "XGB_DIRECT_RICH",
  #   model_group = "direct_rich"
  # )
  # xgb_rich_xai_h$ale <- ensure_xai_columns(xgb_rich_xai_h$ale, kind = "ale") %>%
  #   mutate(component = dplyr::coalesce(component, "forecast_xgb"))
  # xgb_rich_xai_h$interaction <- ensure_xai_columns(xgb_rich_xai_h$interaction, kind = "interaction") %>%
  #   mutate(component = dplyr::coalesce(component, "forecast_xgb"))
  # all_xgb_rich_ale_interaction[[paste0("h", hh)]] <- xgb_rich_xai_h
  # write_csv(xgb_rich_xai_h$ale, paste0("fixed_xgb_direct_rich_ale_h", hh, ".csv"))
  # write_csv(xgb_rich_xai_h$interaction, paste0("fixed_xgb_direct_rich_interaction_h", hh, ".csv"))
  
  # 9.4 SHAP for fixed XGB base
  shap_h <- run_fixed_xgb_shap_one_h(
    data = data,
    predictors = ml_predictors,
    h = hh,
    fit_obj = xgb_run_h$fit
  )
  all_xgb_shap[[paste0("h", hh)]] <- shap_h
  write_csv(shap_h$shap_all, paste0("fixed_xgb_direct_base_shap_h", hh, "_all.csv"))
  write_csv(shap_h$global, paste0("fixed_xgb_direct_base_global_shap_h", hh, ".csv"))
  write_csv(shap_h$local, paste0("fixed_xgb_direct_base_local_shap_h", hh, ".csv"))
  
  xgb_xai_h <- run_fixed_xgb_xai_one_h(
    data = data,
    predictors = ml_predictors,
    h = hh,
    fit_obj = xgb_run_h$fit,
    model_label = "XGB_DIRECT_BASE",
    model_group = "direct"
  )
  xgb_xai_h$ale <- ensure_xai_columns(xgb_xai_h$ale, kind = "ale") %>%
    mutate(component = dplyr::coalesce(component, "forecast_xgb"))
  xgb_xai_h$interaction <- ensure_xai_columns(xgb_xai_h$interaction, kind = "interaction") %>%
    mutate(component = dplyr::coalesce(component, "forecast_xgb"))
  
  all_xgb_ale_interaction[[paste0("h", hh)]] <- xgb_xai_h
  write_csv(xgb_xai_h$ale, paste0("fixed_xgb_direct_base_ale_h", hh, ".csv"))
  write_csv(xgb_xai_h$interaction, paste0("fixed_xgb_direct_base_interaction_h", hh, ".csv"))
  
  # 9.5 TVP vs ML comparison
  tvp_vs_ml_h <- compare_tvp_vs_ml_all_test_dates(
    tvp_beta_last_df = base_run_h$beta_last_test,
    shap_df = shap_h$shap_all,
    h = hh
  )
  all_tvp_vs_ml[[paste0("h", hh)]] <- tvp_vs_ml_h
  write_csv(tvp_vs_ml_h, paste0("fixed_tvp_vs_ml_comparison_h", hh, ".csv"))
  
  # 9.6 residual hybrids
  residual_h_h <- list()
  residual_metrics_h <- list()
  
  for (bm in hybrid_base_models) {
    res_run <- run_all_residual_hybrid_fixed(
      train_base_results = base_run_h$train_results,
      test_base_results = base_run_h$test_results,
      data = data,
      h = hh,
      base_model = bm
    )
    residual_h_h[[bm]] <- res_run
    residual_metrics_h[[bm]] <- res_run$metrics_all
    
    for (v in hybrid_variants) {
      write_csv(
        res_run$runs[[v]]$results,
        paste0("fixed_", tolower(bm), "_residual_hybrid_", v, "_test_h", hh, ".csv")
      )
    }
    write_csv(
      res_run$metrics_all,
      paste0("fixed_", tolower(bm), "_residual_hybrid_metrics_h", hh, ".csv")
    )
  }
  all_residual_hybrid_runs[[paste0("h", hh)]] <- residual_h_h
  residual_shap_h <- run_residual_hybrid_shap_one_h(
    h = hh,
    residual_runs_h = residual_h_h,
    base_models = hybrid_base_models,
    variants = hybrid_variants,
    predictors = ml_predictors
  )
  
  write_csv(
    residual_shap_h$shap_all,
    paste0("fixed_residual_hybrid_shap_h", hh, "_all.csv")
  )
  write_csv(
    residual_shap_h$global,
    paste0("fixed_residual_hybrid_global_shap_h", hh, ".csv")
  )
  write_csv(
    residual_shap_h$local,
    paste0("fixed_residual_hybrid_local_shap_h", hh, ".csv")
  )
  residual_xai_h <- run_residual_hybrid_xai_one_h(
    h = hh,
    residual_runs_h = residual_h_h,
    residual_shap_h = residual_shap_h$shap_all
  )
  
  all_residual_hybrid_xai[[paste0("h", hh)]] <- residual_xai_h
  write_csv(residual_xai_h$full, paste0("fixed_residual_hybrid_xai_full_h", hh, ".csv"))
  write_csv(residual_xai_h$global, paste0("fixed_residual_hybrid_xai_global_h", hh, ".csv"))
  write_csv(residual_xai_h$local, paste0("fixed_residual_hybrid_xai_local_h", hh, ".csv"))
  write_csv(residual_xai_h$shap_link, paste0("fixed_residual_hybrid_xai_shap_link_h", hh, ".csv"))
  
  residual_ale_interaction_h <- run_residual_hybrid_ale_interaction_one_h(
    h = hh,
    residual_runs_h = residual_h_h,
    base_models = hybrid_base_models,
    variants = hybrid_variants,
    predictors = ml_predictors
  )
  residual_ale_interaction_h$ale <- ensure_xai_columns(residual_ale_interaction_h$ale, kind = "ale") %>%
    mutate(component = dplyr::coalesce(component, "residual_pred_ml"))
  residual_ale_interaction_h$interaction <- ensure_xai_columns(residual_ale_interaction_h$interaction, kind = "interaction") %>%
    mutate(component = dplyr::coalesce(component, "residual_pred_ml"))
  all_residual_hybrid_ale_interaction[[paste0("h", hh)]] <- residual_ale_interaction_h
  write_csv(residual_ale_interaction_h$ale, paste0("fixed_residual_hybrid_ale_h", hh, ".csv"))
  write_csv(residual_ale_interaction_h$interaction, paste0("fixed_residual_hybrid_interaction_h", hh, ".csv"))
  
  # 9.7 direct hybrids
  direct_h_h <- list()
  direct_metrics_h <- list()
  
  for (bm in hybrid_base_models) {
    dir_run <- run_all_direct_hybrid_fixed(
      train_base_results = base_run_h$train_results,
      test_base_results = base_run_h$test_results,
      xgb_train_results = xgb_run_h$train_results,
      xgb_test_results = xgb_run_h$test_results,
      data = data,
      h = hh,
      base_model = bm
    )
    direct_h_h[[bm]] <- dir_run
    direct_metrics_h[[bm]] <- dir_run$metrics_all
    
    for (v in names(dir_run$results_by_variant)) {
      write_csv(
        dir_run$results_by_variant[[v]],
        paste0("fixed_", tolower(bm), "_direct_hybrid_", v, "_test_h", hh, ".csv")
      )
    }
    write_csv(
      dir_run$metrics_all,
      paste0("fixed_", tolower(bm), "_direct_hybrid_metrics_h", hh, ".csv")
    )
  }
  all_direct_hybrid_runs[[paste0("h", hh)]] <- direct_h_h
  
  direct_xai_h <- run_direct_hybrid_xai_one_h(
    h = hh,
    direct_runs_h = direct_h_h,
    shap_df_h = shap_h$shap_all
  )
  direct_ale_interaction_h <- run_direct_hybrid_ale_interaction_one_h(
    h = hh,
    direct_runs_h = direct_h_h,
    base_models = hybrid_base_models,
    variants = direct_hybrid_variants,
    predictors = ml_predictors
  )
  if (!exists("all_direct_hybrid_ale_interaction")) {
    all_direct_hybrid_ale_interaction <- list()
  }
  all_direct_hybrid_ale_interaction[[paste0("h", hh)]] <- direct_ale_interaction_h
  
  direct_ale_interaction_h$ale <- ensure_xai_columns(direct_ale_interaction_h$ale, kind = "ale")
  direct_ale_interaction_h$interaction <- ensure_xai_columns(direct_ale_interaction_h$interaction, kind = "interaction")
  all_direct_hybrid_xai[[paste0("h", hh)]] <- direct_xai_h
  write_csv(direct_xai_h$global, paste0("fixed_direct_hybrid_global_xai_h", hh, ".csv"))
  write_csv(direct_xai_h$local, paste0("fixed_direct_hybrid_local_xai_h", hh, ".csv"))
  write_csv(direct_xai_h$ml_dominant_with_shap, paste0("fixed_direct_hybrid_ml_dominant_with_shap_h", hh, ".csv"))
  
  
  write_csv(
    direct_ale_interaction_h$ale,
    paste0("fixed_direct_hybrid_ale_h", hh, ".csv")
  )
  write_csv(
    direct_ale_interaction_h$interaction,
    paste0("fixed_direct_hybrid_interaction_h", hh, ".csv")
  )
  
  save_ale_plots_pdf(
    direct_ale_interaction_h$ale,
    subdir = paste0("direct_h", hh)
  )
  save_interaction_plots_pdf(
    direct_ale_interaction_h$interaction,
    subdir = paste0("direct_h", hh)
  )
  
  save_ale_plots_pdf(
    residual_ale_interaction_h$ale,
    subdir = paste0("residual_h", hh)
  )
  save_interaction_plots_pdf(
    residual_ale_interaction_h$interaction,
    subdir = paste0("residual_h", hh)
  )
  
  save_ale_plots_pdf(
    xgb_xai_h$ale,
    subdir = paste0("xgb_base_h", hh)
  )
  save_interaction_plots_pdf(
    xgb_xai_h$interaction,
    subdir = paste0("xgb_base_h", hh)
  )
  
  # save_ale_plots_pdf(
  #   xgb_rich_xai_h$ale,
  #   subdir = paste0("xgb_rich_h", hh)
  # )
  # save_interaction_plots_pdf(
  #   xgb_rich_xai_h$interaction,
  #   subdir = paste0("xgb_rich_h", hh)
  # )
}

ols_coefs_tbl <- bind_rows(all_ols_coefs)
ar3_coefs_tbl <- bind_rows(all_ar3_coefs)

write_csv(ols_coefs_tbl, "ols_coefficients_all_horizons.csv")
write_csv(ar3_coefs_tbl, "ar3_coefficients_all_horizons.csv")

# =========================================================
# 10. FINAL SUMMARIES
# =========================================================
base_train_summary <- bind_rows(lapply(all_base_runs, `[[`, "train_metrics")) %>%
  arrange(horizon)
base_test_summary <- bind_rows(lapply(all_base_runs, `[[`, "test_metrics")) %>%
  arrange(horizon)

xgb_base_train_summary <- bind_rows(lapply(all_xgb_base_runs, `[[`, "train_metrics")) %>%
  arrange(horizon)
xgb_base_test_summary <- bind_rows(lapply(all_xgb_base_runs, `[[`, "test_metrics")) %>%
  arrange(horizon)

# xgb_rich_train_summary <- bind_rows(lapply(all_xgb_rich_runs, `[[`, "train_metrics")) %>%
#   arrange(horizon)
# xgb_rich_test_summary <- bind_rows(lapply(all_xgb_rich_runs, `[[`, "test_metrics")) %>%
#   arrange(horizon)

residual_hybrid_summary <- bind_rows(lapply(all_residual_hybrid_runs, function(x) {
  bind_rows(lapply(x, `[[`, "metrics_all"))
})) %>% arrange(horizon, base_model, variant)

direct_hybrid_summary <- bind_rows(lapply(all_direct_hybrid_runs, function(x) {
  bind_rows(lapply(x, `[[`, "metrics_all"))
})) %>% arrange(horizon, base_model, variant)

tvp_global_contribution_all_h <- bind_rows(all_tvp_global) %>%
  group_by(horizon) %>%
  mutate(rank_mean_abs_beta_last = min_rank(desc(mean_abs_beta_last))) %>%
  ungroup()

xgb_global_shap_summary_all_h <- bind_rows(lapply(all_xgb_shap, `[[`, "global")) %>%
  group_by(horizon) %>%
  mutate(rank_mean_abs_shap = min_rank(desc(mean_abs_shap))) %>%
  ungroup()

tvp_vs_ml_all_h <- bind_rows(all_tvp_vs_ml)
residual_hybrid_xai_global_all_h <- bind_rows(lapply(all_residual_hybrid_xai, `[[`, "global"))
direct_hybrid_xai_global_all_h <- bind_rows(lapply(all_direct_hybrid_xai, `[[`, "global"))

xgb_ale_all_h <- bind_rows(lapply(all_xgb_ale_interaction, function(x) ensure_xai_columns(x$ale, "ale")))
xgb_interaction_all_h <- bind_rows(lapply(all_xgb_ale_interaction, function(x) ensure_xai_columns(x$interaction, "interaction")))

# xgb_rich_ale_all_h <- bind_rows(lapply(all_xgb_rich_ale_interaction, function(x) ensure_xai_columns(x$ale, "ale")))
# xgb_rich_interaction_all_h <- bind_rows(lapply(all_xgb_rich_ale_interaction, function(x) ensure_xai_columns(x$interaction, "interaction")))

residual_hybrid_ale_all_h <- bind_rows(lapply(all_residual_hybrid_ale_interaction, function(x) ensure_xai_columns(x$ale, "ale")))
residual_hybrid_interaction_all_h <- bind_rows(lapply(all_residual_hybrid_ale_interaction, function(x) ensure_xai_columns(x$interaction, "interaction")))

direct_hybrid_ale_all_h <- bind_rows(lapply(all_direct_hybrid_ale_interaction, function(x) ensure_xai_columns(x$ale, "ale")))
direct_hybrid_interaction_all_h <- bind_rows(lapply(all_direct_hybrid_ale_interaction, function(x) ensure_xai_columns(x$interaction, "interaction")))

write_csv(base_train_summary, "fixed_base_train_summary_all_horizons.csv")
write_csv(base_test_summary, "fixed_base_test_summary_all_horizons.csv")
write_csv(direct_hybrid_ale_all_h, "fixed_direct_hybrid_ale_all_horizons.csv")
write_csv(direct_hybrid_interaction_all_h, "fixed_direct_hybrid_interaction_all_horizons.csv")

# write_csv(xgb_base_train_summary, "fixed_xgb_base_train_summary_all_horizons.csv")
# write_csv(xgb_base_test_summary, "fixed_xgb_base_test_summary_all_horizons.csv")
# write_csv(xgb_rich_train_summary, "fixed_xgb_rich_train_summary_all_horizons.csv")
# write_csv(xgb_rich_test_summary, "fixed_xgb_rich_test_summary_all_horizons.csv")

write_csv(residual_hybrid_summary, "fixed_residual_hybrid_summary_all_horizons.csv")
write_csv(direct_hybrid_summary, "fixed_direct_hybrid_summary_all_horizons.csv")

write_csv(tvp_global_contribution_all_h, "fixed_tvp_global_contribution_all_horizons.csv")
write_csv(xgb_global_shap_summary_all_h, "fixed_xgb_global_shap_summary_all_horizons.csv")
write_csv(tvp_vs_ml_all_h, "fixed_tvp_vs_ml_all_horizons.csv")
write_csv(residual_hybrid_xai_global_all_h, "fixed_residual_hybrid_xai_global_all_horizons.csv")
write_csv(direct_hybrid_xai_global_all_h, "fixed_direct_hybrid_xai_global_all_horizons.csv")
write_csv(xgb_ale_all_h, "fixed_xgb_direct_base_ale_all_horizons.csv")
write_csv(xgb_interaction_all_h, "fixed_xgb_direct_base_interaction_all_horizons.csv")
# write_csv(xgb_rich_ale_all_h, "fixed_xgb_direct_rich_ale_all_horizons.csv")
# write_csv(xgb_rich_interaction_all_h, "fixed_xgb_direct_rich_interaction_all_horizons.csv")
write_csv(residual_hybrid_ale_all_h, "fixed_residual_hybrid_ale_all_horizons.csv")
write_csv(residual_hybrid_interaction_all_h, "fixed_residual_hybrid_interaction_all_horizons.csv")

print(base_test_summary)
print(xgb_base_test_summary)
# print(xgb_rich_test_summary)
print(residual_hybrid_summary)