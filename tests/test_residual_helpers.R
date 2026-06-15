library(testthat)
library(dplyr)

source("residual_gridsearch.R", local = TRUE)

test_that("remove_shock_periods removes purge interval", {

  df <- tibble(
    date = as.Date(c(
      "2020-01-01",
      "2020-02-01",
      "2020-03-01"
    )),
    x = 1:3
  )

  shock_tbl <- tibble(
    purge_start = as.Date("2020-02-01"),
    purge_end = as.Date("2020-02-01")
  )

  out <- remove_shock_periods(df, shock_tbl)

  expect_equal(nrow(out), 2)
  expect_false(any(out$date == as.Date("2020-02-01")))
})

test_that("make_horizon_target creates lead target", {

  df <- tibble(
    date = seq(as.Date("2020-01-01"), by = "month", length.out = 3),
    cpi_mm = c(1, 2, 3)
  )

  out <- make_horizon_target(df, 1)

  expect_equal(out$y_h[1], 2)
  expect_equal(out$y_h[2], 3)
  expect_true(is.na(out$y_h[3]))
})

test_that("compute_decay_weights decreases with age", {

  w <- compute_decay_weights(5, lambda = 0.8)

  expect_equal(length(w), 5)
  expect_gt(w[5], w[1])
})

test_that("calc_rmse works", {

  actual <- c(1, 2, 3)
  pred <- c(1, 2, 4)

  expect_equal(
    calc_rmse(actual, pred),
    sqrt(mean(c(0, 0, 1)))
  )
})

test_that("calc_mae works", {

  actual <- c(1, 2, 3)
  pred <- c(1, 2, 4)

  expect_equal(
    calc_mae(actual, pred),
    mean(c(0, 0, 1))
  )
})
