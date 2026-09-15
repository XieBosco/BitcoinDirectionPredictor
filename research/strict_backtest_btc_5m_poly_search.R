library(tidyverse)
library(lubridate)
library(pROC)

# -------------------------
# Config
# -------------------------
filepath <- "My_Projects/btcusdt_1s_last24h_with_5m_direction.csv"
out_dir <- "research/polymarket_model/r_strict_backtest_poly_results"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Candle-grouped walk-forward settings (same spirit as existing strict script)
min_train_candles <- 220
test_candles <- 48
step_candles <- 24
embargo_candles <- 2

# Polynomial search grid
price_degrees <- c(1, 2, 3, 4)
time_degrees <- c(1, 2, 3, 4)

# -------------------------
# Metrics helpers
# -------------------------
clip_prob <- function(p, eps = 1e-6) {
  pmin(pmax(p, eps), 1 - eps)
}

compute_metrics <- function(y_true, y_prob) {
  y_prob <- clip_prob(y_prob)
  y_pred <- ifelse(y_prob >= 0.5, 1, 0)

  auc_val <- if (length(unique(y_true)) > 1) {
    as.numeric(auc(roc(y_true, y_prob, quiet = TRUE)))
  } else {
    NA_real_
  }

  log_loss <- -mean(y_true * log(y_prob) + (1 - y_true) * log(1 - y_prob))
  brier <- mean((y_prob - y_true)^2)
  acc <- mean(y_pred == y_true)
  calib_gap_pp <- mean(abs(y_prob - y_true)) * 100

  tibble(
    accuracy = acc,
    roc_auc = auc_val,
    log_loss = log_loss,
    brier = brier,
    calibration_gap_pp = calib_gap_pp,
    mean_pred = mean(y_prob),
    mean_actual = mean(y_true)
  )
}

make_candle_splits <- function(candles, min_train = 220, test_size = 48, step = 24, embargo = 2) {
  splits <- list()
  n <- length(candles)
  i <- min_train
  split_id <- 1

  while (i + test_size <= n) {
    train_end <- max(0, i - embargo)
    train_c <- candles[1:train_end]
    test_c <- candles[(i + 1):(i + test_size)]

    if (length(train_c) >= 120 && length(test_c) > 0) {
      splits[[split_id]] <- list(train = train_c, test = test_c)
      split_id <- split_id + 1
    }

    i <- i + step
  }

  splits
}

ewma_vol <- function(r, lambda = 0.94, min_obs = 10) {
  n <- length(r)
  out <- rep(NA_real_, n)
  v <- NA_real_

  for (i in seq_len(n)) {
    ri <- r[i]
    if (!is.finite(ri)) {
      out[i] <- NA_real_
      next
    }

    if (!is.finite(v)) {
      v <- ri^2
    } else {
      v <- lambda * v + (1 - lambda) * (ri^2)
    }

    if (i >= min_obs) {
      out[i] <- sqrt(v)
    }
  }

  out
}

# -------------------------
# Data prep (same core features as existing script)
# -------------------------
btc <- read.csv(filepath) %>%
  mutate(
    direction = case_when(
      candle_5m_direction == "up" ~ 1,
      candle_5m_direction == "down" ~ 0,
      TRUE ~ 1
    ),
    open_time = as.POSIXct(open_time, tz = "UTC"),
    candle_5m = floor_date(open_time, "5 mins")
  ) %>%
  arrange(open_time) %>%
  group_by(candle_5m) %>%
  mutate(
    candle_open = first(open),
    price_change = close - candle_open,
    seconds_since_open = as.numeric(difftime(open_time, candle_5m, units = "secs"))
  ) %>%
  ungroup() %>%
  mutate(
    log_return_1s = log(pmax(close, 1e-12)) - log(pmax(lag(close), 1e-12)),
    ewma_vol = ewma_vol(log_return_1s, lambda = 0.94, min_obs = 10),
    ewma_vol_bps = ewma_vol * 10000,
    buy_volume_ratio = if_else(volume > 0, taker_buy_base_asset_volume / volume, NA_real_)
  ) %>%
  filter(seconds_since_open >= 100, seconds_since_open < 290)

candles <- sort(unique(btc$candle_5m))
splits <- make_candle_splits(
  candles,
  min_train = min_train_candles,
  test_size = test_candles,
  step = step_candles,
  embargo = embargo_candles
)

if (length(splits) == 0) {
  stop("No valid candle-grouped splits. Increase data window or lower min_train_candles.")
}

# -------------------------
# Polynomial search
# -------------------------
combos <- expand.grid(price_degree = price_degrees, time_degree = time_degrees) %>%
  as_tibble() %>%
  arrange(price_degree, time_degree)

all_model_summaries <- list()
all_split_summaries <- list()
all_predictions <- list()

for (ix in seq_len(nrow(combos))) {
  dp <- combos$price_degree[ix]
  dt <- combos$time_degree[ix]
  model_key <- paste0("poly_p", dp, "_t", dt)

  split_metrics <- list()
  split_preds <- list()

  for (k in seq_along(splits)) {
    split_obj <- splits[[k]]

    train_data <- btc %>% filter(candle_5m %in% split_obj$train)
    test_data <- btc %>% filter(candle_5m %in% split_obj$test)

    train_data <- train_data %>%
      filter(
        is.finite(price_change),
        is.finite(seconds_since_open),
        is.finite(ewma_vol_bps),
        is.finite(buy_volume_ratio)
      )
    test_data <- test_data %>%
      filter(
        is.finite(price_change),
        is.finite(seconds_since_open),
        is.finite(ewma_vol_bps),
        is.finite(buy_volume_ratio)
      )

    if (nrow(train_data) < 500 || nrow(test_data) == 0) {
      next
    }

    # Raw polynomial basis keeps this closer to explicit feature engineering.
    f <- as.formula(
      paste0(
        "direction ~ ",
        "poly(price_change, ", dp, ", raw = TRUE) * poly(seconds_since_open, ", dt, ", raw = TRUE) + ",
        "ewma_vol_bps + price_change:ewma_vol_bps + ",
        "buy_volume_ratio + price_change:buy_volume_ratio"
      )
    )

    model <- tryCatch(
      glm(f, data = train_data, family = "binomial"),
      error = function(e) NULL
    )

    if (is.null(model)) {
      next
    }

    preds <- tryCatch(
      predict(model, newdata = test_data, type = "response"),
      error = function(e) rep(NA_real_, nrow(test_data))
    )

    part <- test_data %>%
      transmute(
        model_key = model_key,
        price_degree = dp,
        time_degree = dt,
        split = k,
        open_time,
        candle_5m,
        seconds_since_open,
        direction,
        predicted_prob = preds
      ) %>%
      filter(is.finite(predicted_prob))

    if (nrow(part) == 0) {
      next
    }

    split_preds[[length(split_preds) + 1]] <- part

    m <- compute_metrics(part$direction, part$predicted_prob) %>%
      mutate(
        model_key = model_key,
        price_degree = dp,
        time_degree = dt,
        split = k,
        rows = nrow(part),
        candles = n_distinct(part$candle_5m)
      ) %>%
      select(model_key, price_degree, time_degree, split, rows, candles, everything())

    split_metrics[[length(split_metrics) + 1]] <- m
  }

  if (length(split_preds) == 0) {
    next
  }

  pred_df <- bind_rows(split_preds)
  split_df <- bind_rows(split_metrics)
  overall <- compute_metrics(pred_df$direction, pred_df$predicted_prob) %>%
    mutate(
      model_key = model_key,
      price_degree = dp,
      time_degree = dt,
      rows = nrow(pred_df),
      candles = n_distinct(pred_df$candle_5m),
      splits = n_distinct(pred_df$split)
    ) %>%
    select(model_key, price_degree, time_degree, rows, candles, splits, everything())

  all_predictions[[length(all_predictions) + 1]] <- pred_df
  all_split_summaries[[length(all_split_summaries) + 1]] <- split_df
  all_model_summaries[[length(all_model_summaries) + 1]] <- overall
}

if (length(all_model_summaries) == 0) {
  stop("No model combinations completed successfully.")
}

overall_df <- bind_rows(all_model_summaries) %>%
  arrange(log_loss, desc(roc_auc), brier)

split_df <- bind_rows(all_split_summaries)
pred_df <- bind_rows(all_predictions)

best_key <- overall_df$model_key[1]
best_pred_df <- pred_df %>% filter(model_key == best_key)

# Time-bucket diagnostics for best model
best_bucket_metrics <- best_pred_df %>%
  mutate(
    time_bucket = cut(
      seconds_since_open,
      breaks = c(100, 130, 160, 190, 220, 250, 290),
      right = FALSE,
      include.lowest = TRUE,
      labels = c("100-129", "130-159", "160-189", "190-219", "220-249", "250-289")
    )
  ) %>%
  group_by(time_bucket) %>%
  group_modify(~ compute_metrics(.x$direction, .x$predicted_prob)) %>%
  ungroup()

# Save outputs
write.csv(overall_df, file.path(out_dir, "model_ranking.csv"), row.names = FALSE)
write.csv(split_df, file.path(out_dir, "split_metrics_all_models.csv"), row.names = FALSE)
write.csv(pred_df, file.path(out_dir, "row_level_predictions_all_models.csv"), row.names = FALSE)
write.csv(best_bucket_metrics, file.path(out_dir, "best_model_bucket_metrics.csv"), row.names = FALSE)

cat("=== STRICT POLYNOMIAL SEARCH (R) ===\n")
cat("Combinations tried:", nrow(combos), "\n")
cat("Combinations completed:", nrow(overall_df), "\n\n")

cat("Top 10 models by log_loss:\n")
print(head(overall_df, 10))

cat("\nBest model key:", best_key, "\n")

cat("\nSaved:\n")
cat("-", file.path(out_dir, "model_ranking.csv"), "\n")
cat("-", file.path(out_dir, "split_metrics_all_models.csv"), "\n")
cat("-", file.path(out_dir, "row_level_predictions_all_models.csv"), "\n")
cat("-", file.path(out_dir, "best_model_bucket_metrics.csv"), "\n")
