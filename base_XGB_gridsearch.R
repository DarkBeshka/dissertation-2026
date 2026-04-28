library(tidyverse)
library(lubridate)
library(xgboost)

# =========================================================
# 0. SETTINGS
# =========================================================
train_start <- as.Date("2001-01-01")
train_end   <- as.Date("2023-12-01")
test_start  <- as.Date("2024-01-01")

shock_periods <- tibble::tribble(
  ~start,          ~end,
  # as.Date("2008-09-01"), as.Date("2009-03-01"),
  as.Date("2014-12-01"), as.Date("2015-03-01"),
  # as.Date("2020-03-01"), as.Date("2020-04-01"),
  as.Date("2022-01-01"), as.Date("2022-06-01")
)

horizons <- c(12)#, 3, 6, 12)

max_lag_months <- 12
max_horizon_months <- max(horizons)

ml_predictors <- c(
  "cpi_lag1", "cpi_lag2", "cpi_lag3", "cpi_lag12",
  "m2_log_mom",
  "usd_rub_log_mom",
  "brent_log_mom",
  "business_climate_cbr",
  "unemployment_rate",
  "ppi_construction_mm",
  "ffpi_food_log_mom"
)

# direct XGB defaults
min_xgb_obs <- 48
xgb_nrounds <- 150
xgb_eta <- 0.03
xgb_subsample <- 0.8
xgb_colsample_bytree <- 0.8
xgb_max_depth_default <- 3L

# new regularization defaults
xgb_min_child_weight <- 5
xgb_gamma <- 0.1
xgb_lambda <- 2
xgb_alpha <- 0.1
xgb_max_delta_step <- 0

learning_curves_dir <- "xgb_learning_curves"

# =========================================================
# 1. GRID FOR STANDALONE XGBOOST ONLY
# =========================================================
# Внимание: полная сетка может быть очень большой.
# При необходимости сначала сузьте диапазоны.

# xgb_grid <- tidyr::expand_grid(
#   max_depth = c(2L, 3L),
#   nrounds = c(100L, 200L),
#   eta = c(0.01, 0.02, 0.03),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.5, 0.7, 0.8),
#   min_child_weight = c(7, 10, 15),
#   gamma = c(0.1, 0.3, 0.5, 1),
#   lambda = c(2, 5, 10),
#   alpha = c(0.1, 0.5, 1)
#   #max_delta_step = c(0, 1, 3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# !!!!СОХРАНЯЕМ h1
# xgb_grid <- tidyr::expand_grid(
#   max_depth = c(2L),
#   nrounds = c(160L),  #точка после которой на test начинается деградация
#   eta = c(0.01),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.5),
#   min_child_weight = c(15),
#   gamma = c(1),
#   lambda = c(10),
#   alpha = c(0.1),
#   max_delta_step = c(0)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# !!!! СОХРАНЯЕМ h3
# xgb_grid <- tidyr::expand_grid(
#   max_depth = c(2L),
#   nrounds = c(1000L),
#   eta = c(0.005),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.8),
#   min_child_weight = c(10),
#   gamma = c(1.2),
#   lambda = c(60),
#   alpha = c(3),
#   max_delta_step = c(3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# !!!!! СОХРАНЯЕМ h6, h12
# xgb_grid <- tidyr::expand_grid(
#   max_depth = c(2L),
#   nrounds = c(700L),
#   eta = c(0.008),
#   #subsample = c(0.7, 0.8),
#   colsample_bytree = c(0.8),
#   min_child_weight = c(15),
#   gamma = c(1),
#   lambda = c(40),
#   alpha = c(6),
#   max_delta_step = c(3)
# ) %>%
#   mutate(config_id = row_number()) %>%
#   select(config_id, everything())

# =========================================================
# 2. DATA
# =========================================================
data <- readr::read_csv("dataset_for_model_long.csv", show_col_types = FALSE) %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

data <- data %>%
  arrange(date) %>%
  mutate(
    cpi_lag12 = dplyr::lag(cpi_mm, 12),
    cpi_seasdiff_12 = cpi_mm - dplyr::lag(cpi_mm, 12)
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
# 3. HELPERS
# =========================================================
make_horizon_target <- function(df, h) {
  df %>%
    arrange(date) %>%
    mutate(y_h = dplyr::lead(cpi_mm, h))
}

calc_rmse <- function(actual, pred) {
  sqrt(mean((actual - pred)^2, na.rm = TRUE))
}

calc_mae <- function(actual, pred) {
  mean(abs(actual - pred), na.rm = TRUE)
}

build_learning_curve_manual <- function(fit, train_cc, test_cc, predictors,
                                        config_id, horizon, nrounds) {
  if (is.null(fit) || nrow(train_cc) == 0 || nrow(test_cc) == 0) {
    return(tibble())
  }
  
  x_train <- as.matrix(train_cc %>% select(all_of(predictors)))
  y_train <- train_cc$y_h
  
  x_test <- as.matrix(test_cc %>% select(all_of(predictors)))
  y_test <- test_cc$y_h
  
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
    select(config_id, horizon, iter, sample, metric, value)
}

# prepare_learning_curve_df <- function(evaluation_log, config_id, horizon) {
#   if (is.null(evaluation_log) || nrow(evaluation_log) == 0) {
#     return(tibble())
#   }
#   
#   evaluation_log %>%
#     as_tibble() %>%
#     mutate(
#       config_id = config_id,
#       horizon = horizon
#     ) %>%
#     pivot_longer(
#       cols = -c(iter, config_id, horizon),
#       names_to = "series",
#       values_to = "value"
#     ) %>%
#     mutate(
#       sample = case_when(
#         str_detect(series, "^train_") ~ "train",
#         str_detect(series, "^test_")  ~ "test",
#         TRUE ~ NA_character_
#       ),
#       metric = case_when(
#         str_detect(series, "rmse$") ~ "RMSE",
#         str_detect(series, "mae$")  ~ "MAE",
#         TRUE ~ NA_character_
#       )
#     ) %>%
#     select(config_id, horizon, iter, sample, metric, value)
# }

save_learning_curve_png <- function(curve_df, config_id, horizon, out_dir) {
  if (nrow(curve_df) == 0) {
    message("No learning curve data: config_id=", config_id, ", horizon=", horizon)
    return(invisible(NULL))
  }
  
  metrics <- unique(na.omit(curve_df$metric))
  
  for (m in metrics) {
    plot_df <- curve_df %>% filter(metric == m)
    
    if (nrow(plot_df) == 0) next
    
    p <- ggplot(plot_df, aes(x = iter, y = value, color = sample)) +
      geom_line(linewidth = 0.8) +
      labs(
        title = paste0("Learning curve: ", m,
                       " | config_id=", config_id,
                       " | horizon=", horizon),
        x = "Boosting iteration",
        y = m,
        color = "Sample"
      ) +
      theme_minimal()
    
    file_name <- file.path(
      out_dir,
      paste0(
        "xgb_learning_curve_config_",
        config_id,
        "_h_",
        horizon,
        "_",
        tolower(m),
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

prepare_direct_xgb_data_single <- function(df, predictors, target = "y_h") {
  cols_needed <- unique(c("date", predictors, target))
  cc <- df %>%
    select(all_of(cols_needed)) %>%
    tidyr::drop_na()
  
  if (nrow(cc) == 0) return(NULL)
  
  list(
    cc = cc,
    x = as.matrix(cc %>% select(all_of(predictors))),
    y = cc[[target]]
  )
}

fit_xgb_direct_fixed <- function(train_df, test_df, predictors,
                                 min_obs = min_xgb_obs,
                                 max_depth = xgb_max_depth_default,
                                 nrounds = xgb_nrounds,
                                 eta = xgb_eta,
                                 colsample_bytree = xgb_colsample_bytree,
                                 min_child_weight = xgb_min_child_weight,
                                 gamma = xgb_gamma,
                                 lambda = xgb_lambda,
                                 alpha = xgb_alpha,
                                 max_delta_step = xgb_max_delta_step) {
  prep_train <- prepare_direct_xgb_data_single(train_df, predictors, target = "y_h")
  if (is.null(prep_train)) {
    return(list(
      fit = NULL,
      train_n = 0,
      train_cc = tibble(),
      test_cc = tibble(),
      evaluation_log = tibble(),
      status_reason = "prep_train_is_null"
    ))
  }
  
  if (nrow(prep_train$x) < min_obs) {
    return(list(
      fit = NULL,
      train_n = nrow(prep_train$x),
      train_cc = prep_train$cc,
      test_cc = tibble(),
      evaluation_log = tibble(),
      status_reason = "train_less_than_min_obs"
    ))
  }
  
  prep_test <- prepare_direct_xgb_data_single(test_df, predictors, target = "y_h")
  if (is.null(prep_test) || nrow(prep_test$x) == 0) {
    return(list(
      fit = NULL,
      train_n = nrow(prep_train$x),
      train_cc = prep_train$cc,
      test_cc = tibble(),
      evaluation_log = tibble(),
      status_reason = "prep_test_is_null_or_empty"
    ))
  }
  
  dtrain <- xgb.DMatrix(data = prep_train$x, label = prep_train$y)
  dtest  <- xgb.DMatrix(data = prep_test$x, label = prep_test$y)
  
  params <- list(
    objective = "reg:squarederror",
    eta = eta,
    max_depth = max_depth,
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
    evals = list(train = dtrain, test = dtest),
    verbose = 0
  )
  
  list(
    fit = fit,
    train_n = nrow(prep_train$x),
    train_cc = prep_train$cc,
    test_cc = prep_test$cc,
    evaluation_log = fit$evaluation_log,
    status_reason = "ok"
  )
}

predict_xgb_fixed <- function(fit, full_df, predictors, forecast_col_name = "forecast_xgb") {
  out <- full_df %>%
    select(date, y_h) %>%
    mutate(pred = NA_real_)
  
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

run_fixed_xgb_one_h <- function(data, predictors, h,
                                model_name = "XGB_DIRECT_BASE",
                                min_obs = min_xgb_obs,
                                max_depth = xgb_max_depth_default,
                                nrounds = xgb_nrounds,
                                eta = xgb_eta,
                                #subsample = xgb_subsample,
                                colsample_bytree = xgb_colsample_bytree,
                                min_child_weight = xgb_min_child_weight,
                                gamma = xgb_gamma,
                                lambda = xgb_lambda,
                                alpha = xgb_alpha,
                                max_delta_step = xgb_max_delta_step) {
  df_h <- make_horizon_target(data, h) %>%
    arrange(date)
  
  train_df <- df_h %>% filter(date >= train_start, date <= train_end)
  test_df  <- df_h %>% filter(date >= test_start)
  
  fit_obj <- fit_xgb_direct_fixed(
    train_df = train_df,
    test_df = test_df,
    predictors = predictors,
    min_obs = min_obs,
    max_depth = max_depth,
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
  
  train_results <- predict_xgb_fixed(fit_obj$fit, train_df, predictors, "forecast_xgb") %>%
    mutate(horizon = h, xgb_train_n = fit_obj$train_n) %>%
    arrange(forecast_origin)
  
  test_results <- predict_xgb_fixed(fit_obj$fit, test_df, predictors, "forecast_xgb") %>%
    mutate(horizon = h, xgb_train_n = fit_obj$train_n) %>%
    arrange(forecast_origin)
  
  train_metrics <- compute_xgb_direct_metrics(
    train_results,
    model_name = paste0(model_name, "_TRAIN"),
    h = h
  ) %>%
    mutate(sample = "train")
  
  test_metrics <- compute_xgb_direct_metrics(
    test_results,
    model_name = model_name,
    h = h
  ) %>%
    mutate(sample = "test")
  
  list(
    train_results = train_results,
    test_results = test_results,
    train_metrics = train_metrics,
    test_metrics = test_metrics,
    fit = fit_obj$fit,
    train_n = fit_obj$train_n,
    evaluation_log = fit_obj$evaluation_log,
    train_cc = fit_obj$train_cc,
    test_cc = fit_obj$test_cc,
    status_reason = fit_obj$status_reason
  )
}

append_config_to_metrics <- function(metrics_df, config_row) {
  bind_cols(metrics_df, config_row[rep(1, nrow(metrics_df)), , drop = FALSE]) %>%
    select(
      config_id,
      horizon,
      sample,
      model,
      n_forecasts,
      mae,
      rmse,
      bias,
      max_depth,
      nrounds,
      eta,
      #subsample,
      colsample_bytree,
      min_child_weight,
      gamma,
      lambda,
      alpha,
      max_delta_step,
      everything()
    )
}

run_xgb_grid_search <- function(data,
                                predictors,
                                grid,
                                horizons,
                                min_obs = min_xgb_obs,
                                out_train_file = "xgb_direct_grid_train_metrics.csv",
                                out_test_file = "xgb_direct_grid_test_metrics.csv",
                                learning_curves_dir = "xgb_learning_curves") {
  train_metrics_all <- list()
  test_metrics_all <- list()
  
  for (i in seq_len(nrow(grid))) {
    cfg <- grid[i, , drop = FALSE]
    
    message("====================================================")
    message(
      "Grid config ", i, " / ", nrow(grid),
      " | config_id=", cfg$config_id,
      " | max_depth=", cfg$max_depth,
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
      run_obj <- tryCatch(
        run_fixed_xgb_one_h(
          data = data,
          predictors = predictors,
          h = hh,
          model_name = "XGB_DIRECT_BASE",
          min_obs = min_obs,
          max_depth = cfg$max_depth[[1]],
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
              " failed: ",
              conditionMessage(e)
            )
          )
          NULL
        }
      )
      message(
        "horizon=", hh,
        " | fit_is_null=", is.null(run_obj$fit),
        " | eval_log_rows=", ifelse(is.null(run_obj$evaluation_log), NA, nrow(run_obj$evaluation_log)),
        " | status_reason=", run_obj$status_reason
      )
      
      if (is.null(run_obj)) {
        failed_row_train <- tibble(
          horizon = hh,
          model = "XGB_DIRECT_BASE_TRAIN",
          n_forecasts = NA_integer_,
          mae = NA_real_,
          rmse = NA_real_,
          bias = NA_real_,
          sample = "train",
          train_n = NA_integer_,
          status = "failed"
        )
        
        failed_row_test <- tibble(
          horizon = hh,
          model = "XGB_DIRECT_BASE",
          n_forecasts = NA_integer_,
          mae = NA_real_,
          rmse = NA_real_,
          bias = NA_real_,
          sample = "test",
          train_n = NA_integer_,
          status = "failed"
        )
        
        train_metrics_all[[length(train_metrics_all) + 1]] <-
          append_config_to_metrics(failed_row_train, cfg)
        test_metrics_all[[length(test_metrics_all) + 1]] <-
          append_config_to_metrics(failed_row_test, cfg)
        
        next
      }
      
      curve_wide_df <- build_learning_curve_manual(
        fit = run_obj$fit,
        train_cc = run_obj$train_cc,
        test_cc = run_obj$test_cc,
        predictors = predictors,
        config_id = cfg$config_id[[1]],
        horizon = hh,
        nrounds = cfg$nrounds[[1]]
      )
      
      curve_df <- prepare_learning_curve_df_manual(curve_wide_df)
      
      save_learning_curve_png(
        curve_df = curve_df,
        config_id = cfg$config_id[[1]],
        horizon = hh,
        out_dir = learning_curves_dir
      )
      
      train_metrics_cfg <- run_obj$train_metrics %>%
        mutate(train_n = run_obj$train_n, status = "ok") %>%
        append_config_to_metrics(cfg)
      
      test_metrics_cfg <- run_obj$test_metrics %>%
        mutate(train_n = run_obj$train_n, status = "ok") %>%
        append_config_to_metrics(cfg)
      
      train_metrics_all[[length(train_metrics_all) + 1]] <- train_metrics_cfg
      test_metrics_all[[length(test_metrics_all) + 1]] <- test_metrics_cfg
    }
  }
  
  train_metrics_df <- bind_rows(train_metrics_all) %>%
    arrange(config_id, horizon)
  
  test_metrics_df <- bind_rows(test_metrics_all) %>%
    arrange(config_id, horizon)
  
  readr::write_csv(train_metrics_df, out_train_file)
  readr::write_csv(test_metrics_df, out_test_file)
  
  list(
    train_metrics = train_metrics_df,
    test_metrics = test_metrics_df
  )
}

if (!dir.exists(learning_curves_dir)) {
  dir.create(learning_curves_dir, recursive = TRUE)
}

# =========================================================
# 4. RUN GRID SEARCH
# =========================================================
grid_search_results <- run_xgb_grid_search(
  data = data,
  predictors = ml_predictors,
  grid = xgb_grid,
  horizons = horizons,
  min_obs = min_xgb_obs,
  out_train_file = "xgb_direct_grid_train_metrics.csv",
  out_test_file = "xgb_direct_grid_test_metrics.csv",
  learning_curves_dir = learning_curves_dir
)

# =========================================================
# 5. OPTIONAL SUMMARY TABLES
# =========================================================
best_test_configs <- grid_search_results$test_metrics %>%
  filter(status == "ok") %>%
  group_by(horizon) %>%
  arrange(rmse, .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup()

readr::write_csv(best_test_configs, "xgb_direct_grid_best_test_configs.csv")

best_train_configs <- grid_search_results$train_metrics %>%
  filter(status == "ok") %>%
  group_by(horizon) %>%
  arrange(rmse, .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  ungroup()

readr::write_csv(best_train_configs, "xgb_direct_grid_best_train_configs.csv")

message("GridSearch for standalone XGBoost completed.")
message("Saved files:")
message(" - xgb_direct_grid_train_metrics.csv")
message(" - xgb_direct_grid_test_metrics.csv")
message(" - xgb_direct_grid_best_test_configs.csv")
message(" - xgb_direct_grid_best_train_configs.csv")
message(" - folder with learning curves PNG: ", learning_curves_dir)
message(" - ", file.path(learning_curves_dir, "xgb_learning_curves_all.csv"))