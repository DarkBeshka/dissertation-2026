library(testthat)

source("dieblod_mariano.R", local = TRUE)

test_that("extract_results returns expected columns", {

  fake_dm <- list(
    A = list(
      statistic = 1.5,
      p.value = 0.10
    ),
    B = list(
      statistic = -2,
      p.value = 0.03
    )
  )

  res <- extract_results(
    fake_dm,
    "BASE"
  )

  expect_equal(
    names(res),
    c(
      "Model_1",
      "Model_2",
      "DM_stat",
      "p_value"
    )
  )

  expect_equal(nrow(res), 2)
  expect_equal(res$Model_1[1], "BASE")
})
