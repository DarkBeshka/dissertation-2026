library(tidyverse)
library(lubridate)
library(shrinkTVP)
library(coda)
library(xgboost)
library(forecast)

# =========================================================
# 0. SETTINGS
# =========================================================
train_start <- as.Date("2001-01-01")
train_end   <- as.Date("2023-12-01")
test_start  <- as.Date("2024-01-01")

horizons <- c(1)

# TVP
niter_tvp <- 15000
nburn_tvp <- 7000

tvp_predictors <- c(
  "cpi_lag1", "cpi_lag2", "cpi_lag3",# "cpi_lag12",
  "usd_rub_log_mom",
  "business_climate_cbr",
  "unemployment_rate",
  paste0("month_", 2:12)
)

ml_predictors <- c(
  "cpi_lag1", "cpi_lag2", "cpi_lag3",# "cpi_lag12",
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

# residual hybrids
min_ml_obs <- 48
xgb_nrounds <- 150
xgb_eta <- 0.03
#xgb_subsample <- 0.8
xgb_colsample_bytree <- 0.8
xgb_max_depth_default <- 3L
xgb_max_depth_restricted <- 2L
recency_decay_lambda <- 0.8
ml_window_years <- 10

hybrid_variants <- c("last10y", "weighted", "shallow", "unrestricted")
hybrid_base_models <- c("TVP", "AR3", "OLS")

# дополнительные XGB-параметры для GridSearch
# если хотите строго 1-в-1 поведение fixed_train.R, оставьте эти дефолты
# и не варьируйте их в сетке
xgb_min_child_weight <- 1
xgb_gamma <- 0
xgb_lambda <- 1
xgb_alpha <- 0
xgb_max_delta_step <- 0

shock_periods <- tibble::tribble(
  ~start,          ~end,
  as.Date("2014-12-01"), as.Date("2015-03-01"),
  as.Date("2022-01-01"), as.Date("2022-06-01")
)

max_lag_months <- 12
max_horizon_months <- max(horizons)

learning_curves_dir <- "residual_hybrid_learning_curves"

# =========================================================
# 1. GRID FOR RESIDUAL HYBRIDS
# =========================================================
# Пример полной сетки:
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(3L, 4L),
#   max_depth_restricted = c(2L),
#   nrounds = c(100L, 150L, 300L),
#   eta = c(0.01, 0.03),
#   subsample = c(0.8),
#   colsample_bytree = c(0.5, 0.8),
#   min_child_weight = c(1, 5),
#   gamma = c(0, 0.1, 0.5),
#   lambda = c(1, 2, 5),
#   alpha = c(0, 0.1, 1),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

#!!!! СОХРАНЯЕМ h=1 OLS+ML
residual_xgb_grid <- tidyr::expand_grid(
  max_depth_default = c(2L),
  max_depth_restricted = c(2L),
  #max_depth = c(2L),
  nrounds = c(400L),  #точка после которой на test начинается деградация
  eta = c(0.02),
  #subsample = c(0.7, 0.8),
  colsample_bytree = c(0.4),
  min_child_weight = c(5),
  gamma = c(0.2),
  lambda = c(0.1),
  alpha = c(0),
  max_delta_step = c(0)
) %>%
  mutate(config_id = row_number()) %>%
  select(config_id, everything())

#!!!! СОХРАНЯЕМ h=1 AR+ML
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.02),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.4),
#   min_child_weight = c(5),
#   gamma = c(0.2),
#   lambda = c(0.1),
#   alpha = c(0),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

#!!!! СОХРАНЯЕМ h=1 TVP+ML
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(450L),  #точка после которой на test начинается деградация
#   eta = c(0.02),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.2),
#   min_child_weight = c(5),
#   gamma = c(0.4),
#   lambda = c(0),
#   alpha = c(0),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

#!!!! СОХРАНЯЕМ h=3 AR+ML
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.01),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.7),
#   min_child_weight = c(15),
#   gamma = c(1),
#   lambda = c(10),
#   alpha = c(1),
#   max_delta_step = c(3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

#!!!! СОХРАНЯЕМ h=3 OLS+ML
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.02),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.4),
#   min_child_weight = c(5),
#   gamma = c(0.2),
#   lambda = c(0.1),
#   alpha = c(0),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

#!!!! СОХРАНЯЕМ h=3 TVP+ML
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.03),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.6),
#   min_child_weight = c(5),
#   gamma = c(0.2),
#   lambda = c(0.5),
#   alpha = c(0),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

#!!!! СОХРАНЯЕМ h=6 TVP+ML
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.01),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.7),
#   min_child_weight = c(15),
#   gamma = c(1.7),
#   lambda = c(300),
#   alpha = c(4),
#   max_delta_step = c(3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())


# !!!! СОХРАНЯЕМ h=6 AR+ML
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.05),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.9),
#   min_child_weight = c(15),
#   gamma = c(0.5),
#   lambda = c(200),
#   alpha = c(5),
#   max_delta_step = c(3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# !!!! СОХРАНЯЕМ h=6 OLS+ML
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.05),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.9),
#   min_child_weight = c(15),
#   gamma = c(0.5),
#   lambda = c(200),
#   alpha = c(5),
#   max_delta_step = c(3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# !!!! СОХРАНЯЕМ h=12 TVP+ML unrestricted
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.05),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.3),
#   min_child_weight = c(10),
#   gamma = c(0.1),
#   lambda = c(200),
#   alpha = c(1),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# !!!! СОХРАНЯЕМ h=12 OLS+ML unrestricted
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(400L),  #точка после которой на test начинается деградация
#   eta = c(0.05),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.3),
#   min_child_weight = c(10),
#   gamma = c(0.1),
#   lambda = c(200),
#   alpha = c(1),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# !!!! СОХРАНЯЕМ h=12 AR+ML shallow
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(800L),  #точка после которой на test начинается деградация
#   eta = c(0.03),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.5),
#   min_child_weight = c(10),
#   gamma = c(0.7),
#   lambda = c(70),
#   alpha = c(0.7),
#   max_delta_step = c(3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# !!!! СОХРАНЯЕМ h=12 AR+ML unrestricted
# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(800L),  #точка после которой на test начинается деградация
#   eta = c(0.03),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.5),
#   min_child_weight = c(10),
#   gamma = c(0.7),
#   lambda = c(70),
#   alpha = c(0.7),
#   max_delta_step = c(3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# residual_xgb_grid <- tidyr::expand_grid(
#   max_depth_default = c(2L),
#   max_depth_restricted = c(2L),
#   #max_depth = c(2L),
#   nrounds = c(450L),  #точка после которой на test начинается деградация
#   eta = c(0.02),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.2),
#   min_child_weight = c(5),
#   gamma = c(0.4),
#   lambda = c(0),
#   alpha = c(0),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# =========================================================
# 2. DATA
# =========================================================
data <- read_csv("dataset_for_model_long.csv", show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

data <- data %>%
  arrange(date) %>%
  mutate(
    cpi_lag12 = lag(cpi_mm, 12),
    cpi_seasdiff_12 = cpi_mm - lag(cpi_mm, 12)
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

data <- remove_shock_periods(data, shock_periods_expanded) %>%
  arrange(date)

# =========================================================
# 3. HELPERS
# =========================================================
make_horizon_target <- function(df, h) {
  df %>%
    arrange(date) %>%
    mutate(y_h = dplyr::lead(cpi_mm, h))
}

compute_decay_weights <- function(n, lambda = recency_decay_lambda) {
  idx <- seq_len(n)
  lambda^(rev(idx) - 1)
}

calc_rmse <- function(actual, pred) {
  sqrt(mean((actual - pred)^2, na.rm = TRUE))
}

calc_mae <- function(actual, pred) {
  mean(abs(actual - pred), na.rm = TRUE)
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

build_learning_curve_manual_residual <- function(fit, train_cc, test_cc, predictors,
                                                 config_id, horizon, base_model, variant,
                                                 nrounds) {
  if (is.null(fit) || nrow(train_cc) == 0 || nrow(test_cc) == 0) {
    return(tibble())
  }
  
  x_train <- as.matrix(train_cc %>% select(all_of(predictors)))
  y_train <- train_cc$residual
  
  x_test <- as.matrix(test_cc %>% select(all_of(predictors)))
  y_test <- test_cc$residual
  
  dtrain <- xgb.DMatrix(data = x_train)
  dtest  <- xgb.DMatrix(data = x_test)
  
  curve_list <- vector("list", nrounds)
  
  for (iter in seq_len(nrounds)) {
    pred_train <- as.numeric(
      predict(fit, dtrain, iterationrange = c(1L, iter))
    )
    pred_test <- as.numeric(
      predict(fit, dtest, iterationrange = c(1L, iter))
    )
    
    curve_list[[iter]] <- tibble(
      config_id = config_id,
      horizon = horizon,
      base_model = base_model,
      variant = variant,
      iter = iter,
      train_rmse = calc_rmse(y_train, pred_train),
      train_mae  = calc_mae(y_train, pred_train),
      test_rmse  = calc_rmse(y_test, pred_test),
      test_mae   = calc_mae(y_test, pred_test)
    )
  }
  
  bind_rows(curve_list)
}

prepare_learning_curve_df_manual <- function(curve_wide_df) {
  if (is.null(curve_wide_df) || nrow(curve_wide_df) == 0) {
    return(tibble())
  }
  
  curve_wide_df %>%
    pivot_longer(
      cols = c(train_rmse, train_mae, test_rmse, test_mae),
      names_to = "series",
      values_to = "value"
    ) %>%
    mutate(
      sample = case_when(
        str_detect(series, "^train_") ~ "train",
        str_detect(series, "^test_") ~ "test",
        TRUE ~ NA_character_
      ),
      metric = case_when(
        str_detect(series, "rmse$") ~ "RMSE",
        str_detect(series, "mae$") ~ "MAE",
        TRUE ~ NA_character_
      )
    ) %>%
    select(config_id, horizon, base_model, variant, iter, sample, metric, value)
}

save_learning_curve_png_residual <- function(curve_df, config_id, horizon,
                                             base_model, variant, out_dir) {
  if (nrow(curve_df) == 0) {
    message(
      "No learning curve data: config_id=", config_id,
      ", horizon=", horizon,
      ", base_model=", base_model,
      ", variant=", variant
    )
    return(invisible(NULL))
  }
  
  metrics <- unique(na.omit(curve_df$metric))
  
  for (m in metrics) {
    plot_df <- curve_df %>% filter(metric == m)
    if (nrow(plot_df) == 0) next
    
    p <- ggplot(plot_df, aes(x = iter, y = value, color = sample)) +
      geom_line(linewidth = 0.8) +
      labs(
        title = paste0(
          "Residual hybrid learning curve: ", m,
          " | config_id=", config_id,
          " | horizon=", horizon,
          " | base=", base_model,
          " | variant=", variant
        ),
        x = "Boosting iteration",
        y = m,
        color = "Sample"
      ) +
      theme_minimal()
    
    file_name <- file.path(
      out_dir,
      paste0(
        "residual_hybrid_learning_curve_config_",
        config_id,
        "_base_", tolower(base_model),
        "_variant_", variant,
        "_h_", horizon,
        "_", tolower(m),
        ".png"
      )
    )
    
    ggsave(
      filename = file_name,
      plot = p,
      width = 8,
      height = 5,
      dpi = 150
    )
    
    message("Saved learning curve: ", file_name)
  }
}

# =========================================================
# 4. TVP HELPERS
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

fit_lm_direct <- function(train_df, formula_obj) {
  train_cc <- model.frame(formula_obj, data = train_df, na.action = na.omit)
  if (nrow(train_cc) < 24) return(NULL)
  lm(formula_obj, data = train_df, na.action = na.omit)
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

predict_tvp_fixed <- function(fit, full_df, predictors) {
  out <- full_df %>%
    select(date, y_h) %>%
    mutate(forecast_tvp = NA_real_)
  
  if (is.null(fit) || nrow(full_df) == 0) {
    return(list(
      results = out %>% rename(actual = y_h, forecast_origin = date)
    ))
  }
  
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
  }
  
  list(
    results = out %>%
      mutate(forecast_tvp = pred_vec) %>%
      rename(actual = y_h, forecast_origin = date)
  )
}

# =========================================================
# 5. BASE MODEL FIT/PREDICT
# =========================================================
run_fixed_base_models_one_h <- function(data, h,
                                        tvp_predictors_use = tvp_predictors,
                                        ols_formula_use = ols_formula,
                                        ar3_formula_use = ar3_formula,
                                        niter_tvp_use = niter_tvp,
                                        nburn_tvp_use = nburn_tvp) {
  df_h <- make_horizon_target(data, h) %>%
    arrange(date)
  
  train_df <- df_h %>% filter(date >= train_start, date <= train_end)
  test_df  <- df_h %>% filter(date >= test_start)
  
  tvp_fit <- fit_tvp_direct(
    train_df,
    predictors = tvp_predictors_use,
    niter = niter_tvp_use,
    nburn = nburn_tvp_use
  )
  tvp_train <- predict_tvp_fixed(tvp_fit, train_df, predictors = tvp_predictors_use)
  tvp_test  <- predict_tvp_fixed(tvp_fit, test_df, predictors = tvp_predictors_use)
  
  ols_fit <- fit_lm_direct(train_df, ols_formula_use)
  ols_train <- predict_lm_fixed(ols_fit, train_df, "forecast_ols")
  ols_test  <- predict_lm_fixed(ols_fit, test_df, "forecast_ols")
  
  ar3_fit <- fit_lm_direct(train_df, ar3_formula_use)
  ar3_train <- predict_lm_fixed(ar3_fit, train_df, "forecast_ar3")
  ar3_test  <- predict_lm_fixed(ar3_fit, test_df, "forecast_ar3")
  
  train_results <- train_df %>%
    select(date, y_h) %>%
    rename(forecast_origin = date, actual = y_h) %>%
    left_join(tvp_train$results %>% select(forecast_origin, forecast_tvp), by = "forecast_origin") %>%
    left_join(ols_train %>% select(forecast_origin, forecast_ols), by = "forecast_origin") %>%
    left_join(ar3_train %>% select(forecast_origin, forecast_ar3), by = "forecast_origin") %>%
    mutate(horizon = h) %>%
    arrange(forecast_origin)
  
  test_results <- test_df %>%
    select(date, y_h) %>%
    rename(forecast_origin = date, actual = y_h) %>%
    left_join(tvp_test$results %>% select(forecast_origin, forecast_tvp), by = "forecast_origin") %>%
    left_join(ols_test %>% select(forecast_origin, forecast_ols), by = "forecast_origin") %>%
    left_join(ar3_test %>% select(forecast_origin, forecast_ar3), by = "forecast_origin") %>%
    mutate(horizon = h) %>%
    arrange(forecast_origin)
  
  list(
    train_results = train_results,
    test_results = test_results,
    fits = list(
      tvp = tvp_fit,
      ols = ols_fit,
      ar3 = ar3_fit
    )
  )
}

# =========================================================
# 6. RESIDUAL HYBRIDS (PIPELINE = fixed_train.R)
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
                                   #subsample = xgb_subsample,
                                   colsample_bytree = xgb_colsample_bytree,
                                   min_child_weight = xgb_min_child_weight,
                                   gamma = xgb_gamma,
                                   lambda = xgb_lambda,
                                   alpha = xgb_alpha,
                                   max_delta_step = xgb_max_delta_step) {
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
    return(list(
      fit = NULL,
      train_n = 0,
      train_cc = tibble(),
      status_reason = "prep_train_is_null_or_empty"
    ))
  }
  
  if (nrow(prep$x) < min_obs) {
    return(list(
      fit = NULL,
      train_n = nrow(prep$x),
      train_cc = prep$cc,
      status_reason = "train_obs_below_min_obs"
    ))
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
    #subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha,
    max_delta_step = max_delta_step,
    eval_metric = c("rmse", "mae")
  )
  
  fit <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    verbose = 0
  )
  
  list(
    fit = fit,
    train_n = nrow(prep$x),
    train_cc = prep$cc,
    status_reason = "ok"
  )
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

run_residual_hybrid_fixed_one_h <- function(train_base_results, test_base_results, data, h,
                                            base_model = c("TVP", "AR3", "OLS"),
                                            variant = c("last10y", "weighted", "shallow", "unrestricted"),
                                            predictors = ml_predictors,
                                            min_obs = min_ml_obs,
                                            max_depth_default = xgb_max_depth_default,
                                            max_depth_restricted = xgb_max_depth_restricted,
                                            nrounds = xgb_nrounds,
                                            eta = xgb_eta,
                                            #subsample = xgb_subsample,
                                            colsample_bytree = xgb_colsample_bytree,
                                            min_child_weight = xgb_min_child_weight,
                                            gamma = xgb_gamma,
                                            lambda = xgb_lambda,
                                            alpha = xgb_alpha,
                                            max_delta_step = xgb_max_delta_step) {
  base_model <- match.arg(base_model)
  variant <- match.arg(variant)
  
  hybrid_train <- build_hybrid_ml_base(train_base_results, data, h, base_model)
  hybrid_test  <- build_hybrid_ml_base(test_base_results, data, h, base_model)
  
  fit_ml <- fit_xgb_residual_fixed(
    train_df = hybrid_train,
    predictors = predictors,
    variant = variant,
    max_depth_default = max_depth_default,
    max_depth_restricted = max_depth_restricted,
    min_obs = min_obs,
    nrounds = nrounds,
    eta = eta,
    #subsample = subsample,
    colsample_bytree = colsample_bytree,
    min_child_weight = min_child_weight,
    gamma = gamma,
    lambda = lambda,
    alpha = alpha,
    max_delta_step = max_delta_step
  )
  
  hybrid_train_pred <- predict_xgb_residual_fixed(
    fit = fit_ml$fit,
    full_df = hybrid_train,
    predictors = predictors
  ) %>%
    mutate(
      ml_train_n = fit_ml$train_n,
      forecast_hybrid = base_forecast + residual_pred_ml
    )
  
  hybrid_test_pred <- predict_xgb_residual_fixed(
    fit = fit_ml$fit,
    full_df = hybrid_test,
    predictors = predictors
  ) %>%
    mutate(
      ml_train_n = fit_ml$train_n,
      forecast_hybrid = base_forecast + residual_pred_ml
    )
  
  train_metrics <- compute_hybrid_metrics(hybrid_train_pred, base_model, variant, h) %>%
    mutate(sample = "train")
  
  test_metrics <- compute_hybrid_metrics(hybrid_test_pred, base_model, variant, h) %>%
    mutate(sample = "test")
  
  test_cc <- prepare_ml_single_dataset(hybrid_test, predictors, target = "residual")
  test_cc <- if (is.null(test_cc)) tibble() else test_cc$cc
  
  list(
    train_results = hybrid_train_pred,
    test_results = hybrid_test_pred,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    fit = fit_ml$fit,
    train_n = fit_ml$train_n,
    train_cc = fit_ml$train_cc,
    test_cc = test_cc,
    status_reason = fit_ml$status_reason,
    hybrid_train_base = hybrid_train,
    hybrid_test_base = hybrid_test
  )
}

append_config_to_residual_metrics <- function(metrics_df, config_row) {
  metrics_df %>%
    mutate(
      config_id = config_row$config_id[[1]],
      max_depth_default = config_row$max_depth_default[[1]],
      max_depth_restricted = config_row$max_depth_restricted[[1]],
      nrounds = config_row$nrounds[[1]],
      eta = config_row$eta[[1]],
      colsample_bytree = config_row$colsample_bytree[[1]],
      min_child_weight = config_row$min_child_weight[[1]],
      gamma = config_row$gamma[[1]],
      lambda = config_row$lambda[[1]],
      alpha = config_row$alpha[[1]],
      max_delta_step = config_row$max_delta_step[[1]]
    ) %>%
    select(
      config_id,
      horizon,
      sample,
      base_model,
      variant,
      n_base,
      mae_base,
      rmse_base,
      bias_base,
      n_hybrid,
      mae_hybrid,
      rmse_hybrid,
      bias_hybrid,
      train_n,
      status,
      max_depth_default,
      max_depth_restricted,
      nrounds,
      eta,
      colsample_bytree,
      min_child_weight,
      gamma,
      lambda,
      alpha,
      max_delta_step,
      everything()
    )
}

# =========================================================
# 7. GRID SEARCH FOR RESIDUAL HYBRIDS
# =========================================================
run_residual_hybrid_grid_search <- function(data,
                                            predictors,
                                            grid,
                                            horizons,
                                            base_models = hybrid_base_models,
                                            variants = hybrid_variants,
                                            min_obs = min_ml_obs,
                                            out_train_file = "residual_hybrid_grid_train_metrics.csv",
                                            out_test_file = "residual_hybrid_grid_test_metrics.csv",
                                            learning_curves_dir = "residual_hybrid_learning_curves") {
  train_metrics_all <- list()
  test_metrics_all <- list()
  learning_curves_all <- list()
  
  for (i in seq_len(nrow(grid))) {
    cfg <- grid[i, , drop = FALSE]
    
    message("====================================================")
    message(
      "Grid config ", i, " / ", nrow(grid),
      " | config_id=", cfg$config_id,
      " | max_depth_default=", cfg$max_depth_default,
      " | max_depth_restricted=", cfg$max_depth_restricted,
      " | nrounds=", cfg$nrounds,
      " | eta=", cfg$eta,
      #" | subsample=", cfg$subsample,
      " | colsample_bytree=", cfg$colsample_bytree,
      " | min_child_weight=", cfg$min_child_weight,
      " | gamma=", cfg$gamma,
      " | lambda=", cfg$lambda,
      " | alpha=", cfg$alpha,
      " | max_delta_step=", cfg$max_delta_step
    )
    
    for (hh in horizons) {
      base_obj <- tryCatch(
        run_fixed_base_models_one_h(data = data, h = hh),
        error = function(e) {
          warning(
            paste0(
              "Base models failed for horizon ", hh,
              ": ", conditionMessage(e)
            )
          )
          NULL
        }
      )
      
      if (is.null(base_obj)) {
        for (bm in base_models) {
          for (vv in variants) {
            failed_row_train <- tibble(
              horizon = hh,
              base_model = bm,
              variant = vv,
              n_base = NA_integer_,
              mae_base = NA_real_,
              rmse_base = NA_real_,
              bias_base = NA_real_,
              n_hybrid = NA_integer_,
              mae_hybrid = NA_real_,
              rmse_hybrid = NA_real_,
              bias_hybrid = NA_real_,
              sample = "train",
              train_n = NA_integer_,
              status = "failed_base_models"
            )
            
            failed_row_test <- failed_row_train %>%
              mutate(sample = "test")
            
            train_metrics_all[[length(train_metrics_all) + 1]] <-
              append_config_to_residual_metrics(failed_row_train, cfg)
            test_metrics_all[[length(test_metrics_all) + 1]] <-
              append_config_to_residual_metrics(failed_row_test, cfg)
          }
        }
        next
      }
      
      for (bm in base_models) {
        for (vv in variants) {
          run_obj <- tryCatch(
            run_residual_hybrid_fixed_one_h(
              train_base_results = base_obj$train_results,
              test_base_results = base_obj$test_results,
              data = data,
              h = hh,
              base_model = bm,
              variant = vv,
              predictors = predictors,
              min_obs = min_obs,
              max_depth_default = cfg$max_depth_default[[1]],
              max_depth_restricted = cfg$max_depth_restricted[[1]],
              nrounds = cfg$nrounds[[1]],
              eta = cfg$eta[[1]],
              #subsample = cfg$subsample[[1]],
              colsample_bytree = cfg$colsample_bytree[[1]],
              min_child_weight = cfg$min_child_weight[[1]],
              gamma = cfg$gamma[[1]],
              lambda = cfg$lambda[[1]],
              alpha = cfg$alpha[[1]],
              max_delta_step = cfg$max_delta_step[[1]]
            ),
            error = function(e) {
              warning(
                paste0(
                  "Config ", cfg$config_id,
                  ", horizon ", hh,
                  ", base_model ", bm,
                  ", variant ", vv,
                  " failed: ",
                  conditionMessage(e)
                )
              )
              NULL
            }
          )
          
          message(
            "horizon=", hh,
            " | base_model=", bm,
            " | variant=", vv,
            " | fit_is_null=", is.null(run_obj$fit),
            " | train_cc_rows=", ifelse(is.null(run_obj$train_cc), NA, nrow(run_obj$train_cc)),
            " | test_cc_rows=", ifelse(is.null(run_obj$test_cc), NA, nrow(run_obj$test_cc)),
            " | status_reason=", ifelse(is.null(run_obj), "run_obj_is_null", run_obj$status_reason)
          )
          
          if (is.null(run_obj)) {
            failed_row_train <- tibble(
              horizon = hh,
              base_model = bm,
              variant = vv,
              n_base = NA_integer_,
              mae_base = NA_real_,
              rmse_base = NA_real_,
              bias_base = NA_real_,
              n_hybrid = NA_integer_,
              mae_hybrid = NA_real_,
              rmse_hybrid = NA_real_,
              bias_hybrid = NA_real_,
              sample = "train",
              train_n = NA_integer_,
              status = "failed"
            )
            
            failed_row_test <- failed_row_train %>%
              mutate(sample = "test")
            
            train_metrics_all[[length(train_metrics_all) + 1]] <-
              append_config_to_residual_metrics(failed_row_train, cfg)
            test_metrics_all[[length(test_metrics_all) + 1]] <-
              append_config_to_residual_metrics(failed_row_test, cfg)
            
            next
          }
          
          curve_wide_df <- build_learning_curve_manual_residual(
            fit = run_obj$fit,
            train_cc = run_obj$train_cc,
            test_cc = run_obj$test_cc,
            predictors = predictors,
            config_id = cfg$config_id[[1]],
            horizon = hh,
            base_model = bm,
            variant = vv,
            nrounds = cfg$nrounds[[1]]
          )
          
          curve_df <- prepare_learning_curve_df_manual(curve_wide_df)
          
          if (nrow(curve_df) > 0) {
            learning_curves_all[[length(learning_curves_all) + 1]] <- curve_df
            
            save_learning_curve_png_residual(
              curve_df = curve_df,
              config_id = cfg$config_id[[1]],
              horizon = hh,
              base_model = bm,
              variant = vv,
              out_dir = learning_curves_dir
            )
          }
          
          train_metrics_cfg <- run_obj$train_metrics %>%
            mutate(train_n = run_obj$train_n, status = run_obj$status_reason) %>%
            append_config_to_residual_metrics(cfg)
          
          test_metrics_cfg <- run_obj$test_metrics %>%
            mutate(train_n = run_obj$train_n, status = run_obj$status_reason) %>%
            append_config_to_residual_metrics(cfg)
          
          train_metrics_all[[length(train_metrics_all) + 1]] <- train_metrics_cfg
          test_metrics_all[[length(test_metrics_all) + 1]] <- test_metrics_cfg
        }
      }
    }
  }
  
  train_metrics_df <- if (length(train_metrics_all) > 0) {
    bind_rows(train_metrics_all) %>%
      arrange(config_id, horizon, base_model, variant)
  } else {
    tibble(
      config_id = integer(),
      horizon = integer(),
      sample = character(),
      base_model = character(),
      variant = character(),
      n_base = integer(),
      mae_base = double(),
      rmse_base = double(),
      bias_base = double(),
      n_hybrid = integer(),
      mae_hybrid = double(),
      rmse_hybrid = double(),
      bias_hybrid = double(),
      train_n = integer(),
      status = character(),
      max_depth_default = integer(),
      max_depth_restricted = integer(),
      nrounds = integer(),
      eta = double(),
      colsample_bytree = double(),
      min_child_weight = double(),
      gamma = double(),
      lambda = double(),
      alpha = double(),
      max_delta_step = double()
    )
  }
  
  test_metrics_df <- if (length(test_metrics_all) > 0) {
    bind_rows(test_metrics_all) %>%
      arrange(config_id, horizon, base_model, variant)
  } else {
    tibble(
      config_id = integer(),
      horizon = integer(),
      sample = character(),
      base_model = character(),
      variant = character(),
      n_base = integer(),
      mae_base = double(),
      rmse_base = double(),
      bias_base = double(),
      n_hybrid = integer(),
      mae_hybrid = double(),
      rmse_hybrid = double(),
      bias_hybrid = double(),
      train_n = integer(),
      status = character(),
      max_depth_default = integer(),
      max_depth_restricted = integer(),
      nrounds = integer(),
      eta = double(),
      colsample_bytree = double(),
      min_child_weight = double(),
      gamma = double(),
      lambda = double(),
      alpha = double(),
      max_delta_step = double()
    )
  }
  
  learning_curves_df <- if (length(learning_curves_all) > 0) {
    bind_rows(learning_curves_all) %>%
      arrange(config_id, horizon, base_model, variant, metric, sample, iter)
  } else {
    tibble(
      config_id = integer(),
      horizon = integer(),
      base_model = character(),
      variant = character(),
      iter = integer(),
      sample = character(),
      metric = character(),
      value = double()
    )
  }
  
  list(
    train_metrics = train_metrics_df,
    test_metrics = test_metrics_df,
    learning_curves = learning_curves_df
  )
}

# =========================================================
# 8. RUN GRID SEARCH
# =========================================================
if (!dir.exists(learning_curves_dir)) {
  dir.create(learning_curves_dir, recursive = TRUE)
}

grid_search_results <- run_residual_hybrid_grid_search(
  data = data,
  predictors = ml_predictors,
  grid = residual_xgb_grid,
  horizons = horizons,
  base_models = hybrid_base_models,
  variants = hybrid_variants,
  min_obs = min_ml_obs,
  out_train_file = "residual_hybrid_grid_train_metrics.csv",
  out_test_file = "residual_hybrid_grid_test_metrics.csv",
  learning_curves_dir = learning_curves_dir
)

# =========================================================
# 9. OPTIONAL SUMMARY TABLES
# =========================================================
best_test_configs <- grid_search_results$test_metrics %>%
  filter(status == "ok") %>%
  group_by(horizon, base_model, variant) %>%
  arrange(rmse_hybrid, .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup()

readr::write_csv(
  best_test_configs,
  "residual_hybrid_grid_best_test_configs.csv"
)

best_train_configs <- grid_search_results$train_metrics %>%
  filter(status == "ok") %>%
  group_by(horizon, base_model, variant) %>%
  arrange(rmse_hybrid, .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup()

readr::write_csv(
  best_train_configs,
  "residual_hybrid_grid_best_train_configs.csv"
)

message("GridSearch for residual hybrids completed.")
message("Saved files:")
message(" - residual_hybrid_grid_train_metrics.csv")
message(" - residual_hybrid_grid_test_metrics.csv")
message(" - residual_hybrid_grid_best_test_configs.csv")
message(" - residual_hybrid_grid_best_train_configs.csv")
message(" - folder with learning curves PNG: ", learning_curves_dir)
message(" - ", file.path(learning_curves_dir, "residual_hybrid_learning_curves_all.csv"))