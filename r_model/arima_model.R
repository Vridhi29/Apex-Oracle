# =====================================================
# Apex-Oracle — ARIMA Model (Auto-ARIMA)
# Callable from Python via subprocess
# Outputs predictions as JSON to stdout
# =====================================================

suppressPackageStartupMessages({
  library(forecast)
  library(jsonlite)
})

# ── Parse command-line arguments ──────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat(toJSON(list(error = "Usage: Rscript arima_model.R <csv_path> [days_ahead]"), auto_unbox = TRUE))
  quit(status = 1)
}

csv_path <- args[1]
days_ahead <- ifelse(length(args) >= 2, as.integer(args[2]), 5)

# ── Load Data ─────────────────────────────────────────
tryCatch({
  data <- read.csv(csv_path)

  # Clean data
  data$Close <- as.numeric(data$Close)
  data$Date <- as.Date(data$Date)
  data <- data[!is.na(data$Close), ]

  if (nrow(data) < 60) {
    cat(toJSON(list(error = "Not enough data for ARIMA (need 60+ rows)"), auto_unbox = TRUE))
    quit(status = 1)
  }

  # ── Create time series ──────────────────────────────
  ts_close <- ts(data$Close, frequency = 252)

  # ── Auto-ARIMA fitting ─────────────────────────────
  model <- auto.arima(ts_close, stepwise = TRUE, approximation = TRUE)

  # ── Forecast ────────────────────────────────────────
  fc <- forecast(model, h = days_ahead)

  # ── Build output ────────────────────────────────────
  result <- list(
    status = "ok",
    model = "ARIMA",
    model_order = paste0("ARIMA(", model$arma[1], ",", model$arma[6], ",", model$arma[2], ")"),
    current_price = as.numeric(tail(data$Close, 1)),
    predictions = as.numeric(fc$mean),
    lower_80 = as.numeric(fc$lower[, 1]),
    upper_80 = as.numeric(fc$upper[, 1]),
    lower_95 = as.numeric(fc$lower[, 2]),
    upper_95 = as.numeric(fc$upper[, 2]),
    aic = as.numeric(AIC(model)),
    bic = as.numeric(BIC(model)),
    next_day_prediction = as.numeric(fc$mean[1]),
    change_pct = as.numeric((fc$mean[1] - tail(data$Close, 1)) / tail(data$Close, 1) * 100),
    direction = ifelse(fc$mean[1] > tail(data$Close, 1), "UP", "DOWN")
  )

  cat(toJSON(result, auto_unbox = TRUE, pretty = TRUE))

}, error = function(e) {
  cat(toJSON(list(
    status = "error",
    model = "ARIMA",
    error = as.character(e$message)
  ), auto_unbox = TRUE))
  quit(status = 1)
})