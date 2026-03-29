# =====================================================
# Apex-Oracle — GARCH Model (Volatility Forecasting)
# Callable from Python via subprocess
# Outputs volatility forecasts as JSON to stdout
# =====================================================

suppressPackageStartupMessages({
  library(rugarch)
  library(jsonlite)
})

# ── Parse command-line arguments ──────────────────────
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  cat(toJSON(list(error = "Usage: Rscript garch_model.R <csv_path> [days_ahead]"), auto_unbox = TRUE))
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

  if (nrow(data) < 100) {
    cat(toJSON(list(error = "Not enough data for GARCH (need 100+ rows)"), auto_unbox = TRUE))
    quit(status = 1)
  }

  # ── Compute returns ──────────────────────────────────
  returns <- diff(log(data$Close))
  returns <- returns[!is.na(returns) & is.finite(returns)]

  # ── Specify GARCH(1,1) model ─────────────────────────
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(1, 0), include.mean = TRUE),
    distribution.model = "std"  # Student-t for fat tails
  )

  # ── Fit model ────────────────────────────────────────
  fit <- ugarchfit(spec, data = returns, solver = "hybrid")

  # ── Forecast ─────────────────────────────────────────
  fc <- ugarchforecast(fit, n.ahead = days_ahead)

  # ── Extract results ──────────────────────────────────
  sigma_forecast <- as.numeric(sigma(fc))
  mean_forecast <- as.numeric(fitted(fc))

  # Convert log-return forecasts to price predictions
  last_price <- tail(data$Close, 1)
  predicted_prices <- last_price * exp(cumsum(mean_forecast))

  # Volatility-based confidence bands
  cumulative_vol <- sqrt(cumsum(sigma_forecast^2))
  lower_band <- last_price * exp(cumsum(mean_forecast) - 1.96 * cumulative_vol)
  upper_band <- last_price * exp(cumsum(mean_forecast) + 1.96 * cumulative_vol)

  # ── Build output ─────────────────────────────────────
  result <- list(
    status = "ok",
    model = "GARCH",
    model_type = "GARCH(1,1) with Student-t",
    current_price = as.numeric(last_price),
    predictions = as.numeric(predicted_prices),
    volatility_forecast = as.numeric(sigma_forecast),
    annualized_volatility = as.numeric(sigma_forecast * sqrt(252)),
    lower_95 = as.numeric(lower_band),
    upper_95 = as.numeric(upper_band),
    next_day_prediction = as.numeric(predicted_prices[1]),
    next_day_volatility = as.numeric(sigma_forecast[1]),
    change_pct = as.numeric((predicted_prices[1] - last_price) / last_price * 100),
    direction = ifelse(predicted_prices[1] > last_price, "UP", "DOWN")
  )

  cat(toJSON(result, auto_unbox = TRUE, pretty = TRUE))

}, error = function(e) {
  cat(toJSON(list(
    status = "error",
    model = "GARCH",
    error = as.character(e$message)
  ), auto_unbox = TRUE))
  quit(status = 1)
})
