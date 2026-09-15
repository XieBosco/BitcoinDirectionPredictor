library(tidyverse)
library(lubridate)
library(pROC)

# -------------------------
# Config
# -------------------------
filepath <- "My_Projects/btcusdt_1s_last24h_with_5m_direction.csv"
out_dir <- "research/polymarket_model/r_strict_backtest_results"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

# Candle-grouped walk-forward settings
min_train_candles <- 220
test_candles <- 48
step_candles <- 24
embargo_candles <- 2

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
# Data prep
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
# Strict walk-forward
# -------------------------
all_preds <- list()
all_split_metrics <- list()

for (k in seq_along(splits)) {
  split_obj <- splits[[k]]

  train_data <- btc %>% filter(candle_5m %in% split_obj$train)
  test_data <- btc %>% filter(candle_5m %in% split_obj$test)

  # Keep model inputs complete for stable training/inference.
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

  # Add EWMA volatility so probabilities can adapt to changing uncertainty.
  model <- glm(
    direction ~
      price_change * seconds_since_open +
      ewma_vol_bps +
      price_change:ewma_vol_bps +
      buy_volume_ratio +
      price_change:buy_volume_ratio,
    data = train_data,
    family = "binomial"
  )
  preds <- predict(model, newdata = test_data, type = "response")

  part <- test_data %>%
    transmute(
      split = k,
      open_time,
      candle_5m,
      seconds_since_open,
      ewma_vol_bps,
      buy_volume_ratio,
      direction,
      predicted_prob = preds
    )

  all_preds[[k]] <- part

  m <- compute_metrics(part$direction, part$predicted_prob) %>%
    mutate(split = k, rows = nrow(part), candles = n_distinct(part$candle_5m)) %>%
    select(split, rows, candles, everything())

  all_split_metrics[[k]] <- m
}

pred_df <- bind_rows(all_preds)
if (nrow(pred_df) == 0) {
  stop("No predictions were produced. Check data quality and feature availability.")
}
split_metrics <- bind_rows(all_split_metrics)
overall <- compute_metrics(pred_df$direction, pred_df$predicted_prob)

# Time-bucket diagnostics
pred_df <- pred_df %>%
  mutate(
    time_bucket = cut(
      seconds_since_open,
      breaks = c(100, 130, 160, 190, 220, 250, 290),
      right = FALSE,
      include.lowest = TRUE,
      labels = c("100-129", "130-159", "160-189", "190-219", "220-249", "250-289")
    )
  )

bucket_metrics <- pred_df %>%
  group_by(time_bucket) %>%
  group_modify(~ compute_metrics(.x$direction, .x$predicted_prob)) %>%
  ungroup() %>%
  mutate(n_rows = pred_df %>% count(time_bucket) %>% pull(n))

# Save outputs
write.csv(pred_df, file.path(out_dir, "row_level_predictions.csv"), row.names = FALSE)
write.csv(split_metrics, file.path(out_dir, "split_metrics.csv"), row.names = FALSE)
write.csv(overall, file.path(out_dir, "overall_metrics.csv"), row.names = FALSE)
write.csv(bucket_metrics, file.path(out_dir, "bucket_metrics.csv"), row.names = FALSE)

cat("=== STRICT CANDLE-GROUPED BACKTEST (R) ===\n")
cat("Rows used:", nrow(pred_df), "\n")
cat("Candles used:", n_distinct(pred_df$candle_5m), "\n")
cat("Splits:", nrow(split_metrics), "\n\n")

cat("Overall:\n")
print(overall)

cat("\nSaved:\n")
cat("-", file.path(out_dir, "row_level_predictions.csv"), "\n")
cat("-", file.path(out_dir, "split_metrics.csv"), "\n")
cat("-", file.path(out_dir, "overall_metrics.csv"), "\n")
cat("-", file.path(out_dir, "bucket_metrics.csv"), "\n")
