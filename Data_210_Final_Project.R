# =============================================================================
# Data 210 - Final Project
# Predicting the Unpredictable: A Quantitative Analysis of NVIDIA Stock
# Author: Ethan Wong
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────

required_packages <- c(
  "here", "dplyr", "tidyr", "ggplot2", "knitr",
  "tseries", "forecast", "TTR", "zoo",
  "randomForest", "glmnet"
)

lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
})

# ── Data Loading ──────────────────────────────────────────────────────────────

nvda_data <- read.csv(here("data", "NVDA_yfinance_clean.csv"))
df <- nvda_data

df$Date     <- as.Date(df$Date)
df$Volume_M <- df$Volume / 1e6

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

# 1. Price-Derived Features
df <- df %>%
  mutate(
    daily_range    = High - Low,
    overnight_gap  = Open - lag(Close),
    close_position = (Close - Low) / (High - Low + 1e-8),
    log_return     = log(Close / lag(Close))
  )

# 2. Rolling and Lagged Features
df <- df %>%
  mutate(
    lag1 = lag(log_return, 1),
    lag2 = lag(log_return, 2),
    lag3 = lag(log_return, 3),
    lag4 = lag(log_return, 4),
    lag5 = lag(log_return, 5),

    vol_5d  = rollapply(log_return, width = 5,  FUN = sd,   fill = NA, align = "right"),
    vol_20d = rollapply(log_return, width = 20, FUN = sd,   fill = NA, align = "right"),

    ma_5d  = rollapply(log_return, width = 5,  FUN = mean, fill = NA, align = "right"),
    ma_20d = rollapply(log_return, width = 20, FUN = mean, fill = NA, align = "right"),

    sma_20         = SMA(Close, n = 20),
    price_vs_sma20 = Close / sma_20 - 1
  )

# 3. Technical Indicators
df$rsi_14 <- RSI(df$Close, n = 14)

macd_vals      <- MACD(df$Close, nFast = 12, nSlow = 26, nSig = 9)
df$macd        <- macd_vals[, "macd"]
df$macd_signal <- macd_vals[, "signal"]
df$macd_hist   <- df$macd - df$macd_signal

bb          <- BBands(df$Close, n = 20, sd = 2)
df$bb_pct_b <- bb[, "pctB"]
df$bb_width <- (bb[, "up"] - bb[, "dn"]) / bb[, "mavg"]

# 4. Volume Features
df <- df %>%
  mutate(
    vol_ma20   = rollapply(Volume, width = 20, FUN = mean, fill = NA, align = "right"),
    rel_volume = Volume / vol_ma20,
    log_volume = log(Volume + 1)
  )

# 5. Binary Target Variable
df <- df %>%
  mutate(
    target = as.integer(lead(Close) > Close)
  )

# 6. Drop NA rows from rolling window warmup
df_model <- df %>% drop_na()

cat("Rows after feature engineering:", nrow(df_model), "\n")
cat("Features available:\n")
print(names(df_model))

# =============================================================================
# EDA — UNIVARIATE
# =============================================================================

# Figure 1: Log-Transformed Price Distributions
par(mfrow = c(2, 2), oma = c(0, 0, 3, 0), mar = c(4, 4, 2, 1))
hist(log(df$Close), main = "Log(Close Price)", xlab = "log(USD)", col = "#76b900", breaks = 60)
hist(log(df$High),  main = "Log(High Price)",  xlab = "log(USD)", col = "#4a90d9", breaks = 60)
hist(log(df$Low),   main = "Log(Low Price)",   xlab = "log(USD)", col = "#e05c5c", breaks = 60)
hist(log(df$Open),  main = "Log(Open Price)",  xlab = "log(USD)", col = "#f4a83a", breaks = 60)
mtext("Figure 1: Log-Transformed Price Distributions", side = 3, outer = TRUE, line = 1, cex = 1.2, font = 2)
par(mfrow = c(1, 1), oma = c(0, 0, 0, 0), mar = c(5, 4, 4, 2))

# Figure 2: Log Return Distribution
hist(df_model$log_return,
     main   = "Figure 2: Distribution of Daily Log Returns",
     xlab   = "Log Return",
     col    = "#2ecc71",
     breaks = 80)

# =============================================================================
# EDA — BIVARIATE
# =============================================================================

# Figure 3: Close Price Over Time
ggplot(df, aes(x = Date, y = Close)) +
  geom_line(color = "#76b900", linewidth = 0.6) +
  labs(title = "Figure 3: NVDA Close Price (2016 to 2025)", x = "Date", y = "Close (USD)") +
  theme_minimal()

# Figure 4: Trading Volume Over Time
ggplot(df, aes(x = Date, y = Volume_M)) +
  geom_line(color = "#9b59b6", linewidth = 0.3, alpha = 0.3) +
  geom_smooth(method = "loess", span = 0.15, se = TRUE,
              color = "#6c3483", fill = "#d2b4de", linewidth = 1.1) +
  labs(
    title   = "Figure 4: NVDA Daily Trading Volume (Millions of Shares), 2016 to 2025",
    x       = "Date",
    y       = "Volume (Millions)",
    caption = "Note: Faded line represents raw daily values. Solid line is a LOESS trend with a 95% confidence interval ribbon."
  ) +
  theme_minimal() +
  theme(
    plot.caption          = element_text(hjust = 0, size = 9, color = "gray30", lineheight = 1.2),
    plot.caption.position = "plot",
    plot.margin           = ggplot2::margin(t = 10, r = 10, b = 20, l = 10)
  )

# Table 4: Annual Volume Summary
df %>%
  mutate(Year = format(Date, "%Y")) %>%
  group_by(Year) %>%
  summarise(
    `Median Vol (M)` = round(median(Volume_M, na.rm = TRUE), 1),
    `Mean Vol (M)`   = round(mean(Volume_M,   na.rm = TRUE), 1),
    `Max Vol (M)`    = round(max(Volume_M,    na.rm = TRUE), 1),
    `Trading Days`   = n()
  ) %>%
  kable(caption = "Table 4: Annual Volume Summary (Millions of Shares)")

# =============================================================================
# EDA — ENGINEERED FEATURES OVER TIME
# =============================================================================

# Figure 5: Log Returns Over Time
df_model <- df_model %>%
  mutate(smooth_vol_upper =  2 * vol_20d,
         smooth_vol_lower = -2 * vol_20d)

ggplot(df_model, aes(x = Date)) +
  geom_ribbon(aes(ymin = smooth_vol_lower, ymax = smooth_vol_upper,
                  fill = "2x 20-Day Vol Envelope"),
              alpha = 0.35) +
  geom_line(aes(y = log_return, color = "Raw Log Return"),
            linewidth = 0.25, alpha = 0.4) +
  geom_smooth(aes(y = log_return, color = "LOESS Trend"),
              method = "loess", span = 0.1, se = FALSE, linewidth = 1.0) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_manual(
    name   = "Series",
    values = c("Raw Log Return" = "#2ecc71", "LOESS Trend" = "#1a7a40")
  ) +
  scale_fill_manual(
    name   = "Envelope",
    values = c("2x 20-Day Vol Envelope" = "#a9dfbf")
  ) +
  labs(
    title   = "Figure 5: NVDA Daily Log Returns Over Time",
    x       = "Date",
    y       = "Log Return",
    caption = "Note: Shaded ribbon represents +/- 2 standard deviations of the 20-day rolling volatility."
  ) +
  theme_minimal() +
  theme(
    legend.position       = "bottom",
    plot.caption          = element_text(hjust = 0, size = 9, color = "gray30"),
    plot.caption.position = "plot"
  )

# Table 5: Annual Log Return Summary
df_model %>%
  mutate(Year = format(Date, "%Y")) %>%
  group_by(Year) %>%
  summarise(
    `Mean Return`   = round(mean(log_return,   na.rm = TRUE), 4),
    `Median Return` = round(median(log_return, na.rm = TRUE), 4),
    `Std Dev`       = round(sd(log_return,     na.rm = TRUE), 4),
    `Max Return`    = round(max(log_return,    na.rm = TRUE), 4),
    `Min Return`    = round(min(log_return,    na.rm = TRUE), 4),
    `Trading Days`  = n()
  ) %>%
  kable(caption = "Table 5: Annual Log Return Summary Statistics")

# Figure 6: Rolling Volatility Over Time
df_model %>%
  tidyr::pivot_longer(cols = c(vol_5d, vol_20d),
                      names_to = "Window", values_to = "Volatility") %>%
  mutate(Window = recode(Window, vol_5d = "5-Day", vol_20d = "20-Day")) %>%
  ggplot(aes(x = Date, y = Volatility, color = Window, alpha = Window, linewidth = Window)) +
  geom_line() +
  scale_color_manual(
    name   = "Rolling Window",
    values = c("5-Day" = "#e67e22", "20-Day" = "#c0392b")
  ) +
  scale_alpha_manual(values = c("5-Day" = 0.35, "20-Day" = 0.9)) +
  scale_linewidth_manual(values = c("5-Day" = 0.3, "20-Day" = 0.8)) +
  labs(
    title   = "Figure 6: Rolling Volatility Over Time (5-Day vs 20-Day)",
    x       = "Date",
    y       = "Std Dev of Log Return",
    caption = "Note: 5-day line is faded to reduce visual noise. 20-day line captures regime-level volatility."
  ) +
  guides(alpha = "none", linewidth = "none") +
  theme_minimal() +
  theme(
    legend.position       = "bottom",
    plot.caption          = element_text(hjust = 0, size = 9, color = "gray30"),
    plot.caption.position = "plot"
  )

# Table 6: Annual 20-Day Rolling Volatility Summary
df_model %>%
  mutate(Year = format(Date, "%Y")) %>%
  group_by(Year) %>%
  summarise(
    `Median 20D Vol` = round(median(vol_20d, na.rm = TRUE), 4),
    `Mean 20D Vol`   = round(mean(vol_20d,   na.rm = TRUE), 4),
    `Max 20D Vol`    = round(max(vol_20d,    na.rm = TRUE), 4)
  ) %>%
  kable(caption = "Table 6: Annual 20-Day Rolling Volatility Summary")

# Figure 7: RSI Over Time
ggplot(df_model, aes(x = Date, y = rsi_14)) +
  annotate("rect", xmin = min(df_model$Date), xmax = max(df_model$Date),
           ymin = 70, ymax = 100, fill = "#fadbd8", alpha = 0.4) +
  annotate("rect", xmin = min(df_model$Date), xmax = max(df_model$Date),
           ymin = 0,  ymax = 30,  fill = "#d6eaf8", alpha = 0.4) +
  geom_line(aes(color = "Raw RSI"), linewidth = 0.25, alpha = 0.4) +
  geom_smooth(aes(color = "LOESS Trend"), method = "loess", span = 0.08,
              se = FALSE, linewidth = 1.0) +
  geom_hline(yintercept = 70, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
  geom_hline(yintercept = 30, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
  geom_hline(yintercept = 50, linetype = "dotted", color = "gray50") +
  scale_color_manual(
    name   = "RSI Series",
    values = c("Raw RSI" = "#3498db", "LOESS Trend" = "#1a5276")
  ) +
  annotate("text", x = min(df_model$Date), y = 73,
           label = "Overbought (70)", hjust = 0, size = 3, color = "#c0392b") +
  annotate("text", x = min(df_model$Date), y = 27,
           label = "Oversold (30)",   hjust = 0, size = 3, color = "#c0392b") +
  labs(
    title   = "Figure 7: 14-Day RSI Over Time",
    x       = "Date",
    y       = "RSI",
    caption = "Note: Red shaded zone marks overbought readings (RSI >= 70). Blue shaded zone marks oversold readings (RSI <= 30)."
  ) +
  theme_minimal() +
  theme(
    legend.position       = "bottom",
    plot.caption          = element_text(hjust = 0, size = 9, color = "gray30"),
    plot.caption.position = "plot"
  )

# Table 7: Days Spent in Each RSI Zone by Year
df_model %>%
  mutate(
    Year   = format(Date, "%Y"),
    Signal = case_when(
      rsi_14 >= 70 ~ "Overbought (>=70)",
      rsi_14 <= 30 ~ "Oversold (<=30)",
      TRUE         ~ "Neutral"
    )
  ) %>%
  group_by(Year, Signal) %>%
  summarise(Days = n(), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = Signal, values_from = Days, values_fill = 0) %>%
  mutate(`Total Days` = rowSums(across(where(is.numeric)))) %>%
  kable(caption = "Table 7: Days Spent in Each RSI Zone by Year")

# Figure 8: MACD Histogram Over Time
ggplot(df_model, aes(x = Date, y = macd_hist, fill = macd_hist > 0)) +
  geom_col(width = 1, alpha = 0.75) +
  scale_fill_manual(
    values = c("TRUE"  = "#27ae60", "FALSE" = "#e74c3c"),
    labels = c("TRUE"  = "Bullish (MACD > Signal)", "FALSE" = "Bearish (MACD < Signal)"),
    name   = "Market Momentum"
  ) +
  geom_smooth(aes(x = Date, y = macd_hist),
              method      = "loess",
              span        = 0.1,
              se          = FALSE,
              color       = "#2c3e50",
              linewidth   = 0.9,
              inherit.aes = FALSE) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  labs(
    title   = "Figure 8: MACD Histogram Over Time",
    x       = "Date",
    y       = "MACD minus Signal",
    caption = "Note: Dark overlaid line is a LOESS smooth of the histogram values."
  ) +
  theme_minimal() +
  theme(
    legend.position       = "bottom",
    plot.caption          = element_text(hjust = 0, size = 9, color = "gray30"),
    plot.caption.position = "plot"
  )

# Table 8: MACD Bullish vs Bearish Days by Year
df_model %>%
  mutate(
    Year   = format(Date, "%Y"),
    Signal = ifelse(macd_hist > 0, "Bullish", "Bearish")
  ) %>%
  group_by(Year, Signal) %>%
  summarise(Days = n(), .groups = "drop") %>%
  tidyr::pivot_wider(names_from = Signal, values_from = Days, values_fill = 0) %>%
  mutate(
    `Total Days`  = Bullish + Bearish,
    `Pct Bullish` = round(100 * Bullish / `Total Days`, 1)
  ) %>%
  kable(caption = "Table 8: MACD Bullish vs Bearish Days by Year")

# =============================================================================
# STATIONARITY TESTING
# =============================================================================

# ADF on raw price -- expect FAIL (non-stationary)
cat("=== ADF Test: Raw Close Price ===\n")
print(adf.test(df$Close))

# ADF on log returns -- expect PASS (stationary)
log_ret_clean <- na.omit(df$log_return)
cat("\n=== ADF Test: Log Returns ===\n")
print(adf.test(log_ret_clean))

# Figure 9: ACF of Log Returns
par(mar = c(5, 4, 4, 2))
Acf(log_ret_clean, main = "Figure 9: ACF of NVDA Log Returns", lag.max = 35)

# Figure 9b: PACF of Log Returns
Pacf(log_ret_clean, main = "Figure 9b: PACF of NVDA Log Returns", lag.max = 35)

# =============================================================================
# CLUSTERING
# =============================================================================

# Figure 10: K-Means Clustering
features        <- df %>% select(Close, High, Low, Open, Volume)
scaled_features <- scale(features)

set.seed(123)
kmeans_result <- kmeans(scaled_features, centers = 3)
df$cluster    <- as.factor(kmeans_result$cluster)

ggplot(df, aes(x = Close, y = Volume, color = cluster)) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(
    title = "Figure 10: K-Means Clustering of NVDA Price and Volume",
    x     = "Close Price",
    y     = "Volume",
    color = "Cluster ID"
  )

# =============================================================================
# MODELING — TRAIN/TEST SPLIT
# =============================================================================

feature_cols <- c("log_return", "lag1", "lag2", "lag3", "lag4", "lag5",
                  "vol_5d", "vol_20d", "ma_5d", "ma_20d", "price_vs_sma20",
                  "rsi_14", "macd_hist", "bb_pct_b", "bb_width",
                  "rel_volume", "log_volume", "daily_range",
                  "overnight_gap", "close_position")

n       <- nrow(df_model)
n_train <- floor(0.8 * n)

train_df <- df_model[1:n_train, ]
test_df  <- df_model[(n_train + 1):n, ]

X_train <- train_df[, feature_cols]
y_train <- as.factor(train_df$target)

X_test  <- test_df[, feature_cols]
y_test  <- as.factor(test_df$target)

X_train_scaled <- scale(X_train)
X_test_scaled  <- scale(X_test,
                        center = attr(X_train_scaled, "scaled:center"),
                        scale  = attr(X_train_scaled, "scaled:scale"))

cat("Training observations:", n_train, "\n")
cat("Test observations:    ", n - n_train, "\n")
cat("Features:             ", length(feature_cols), "\n")
cat("Class balance (train):\n")
print(table(y_train))

# =============================================================================
# EVALUATION HELPER
# =============================================================================

get_metrics <- function(pred, actual, model_name) {
  pred   <- as.character(pred)
  actual <- as.character(actual)
  TP  <- sum(pred == "1" & actual == "1")
  TN  <- sum(pred == "0" & actual == "0")
  FP  <- sum(pred == "1" & actual == "0")
  FN  <- sum(pred == "0" & actual == "1")
  acc <- (TP + TN) / (TP + TN + FP + FN)
  pre <- if ((TP + FP) > 0) TP / (TP + FP) else NA
  rec <- if ((TP + FN) > 0) TP / (TP + FN) else NA
  f1  <- if (!is.na(pre) && !is.na(rec) && (pre + rec) > 0) {
           2 * pre * rec / (pre + rec)
         } else NA
  data.frame(
    Model     = model_name,
    Accuracy  = round(acc, 4),
    Precision = round(pre, 4),
    Recall    = round(rec, 4),
    F1        = round(f1,  4),
    row.names = NULL
  )
}

results_list <- list()

# =============================================================================
# MODEL 1: ARIMA
# =============================================================================

train_returns <- train_df$log_return
test_returns  <- test_df$log_return

set.seed(42)
arima_model <- auto.arima(train_returns, seasonal = FALSE, stepwise = TRUE,
                          approximation = FALSE)

cat("=== ARIMA Model Selected ===\n")
print(summary(arima_model))

# Residual diagnostics
checkresiduals(arima_model)

# Rolling directional forecast
arima_order   <- arimaorder(arima_model)
p_a <- arima_order[1]; d_a <- arima_order[2]; q_a <- arima_order[3]

all_returns   <- c(train_returns, test_returns)
n_test        <- length(test_returns)
rolling_preds <- numeric(n_test)

for (i in seq_len(n_test)) {
  history          <- all_returns[1:(n_train + i - 1)]
  fit_i            <- Arima(history, order = c(p_a, d_a, q_a))
  rolling_preds[i] <- as.numeric(forecast(fit_i, h = 1)$mean)
}

pred_arima <- as.factor(ifelse(rolling_preds >= 0, 1, 0))
results_list[["ARIMA"]] <- get_metrics(pred_arima, y_test, "ARIMA")

# =============================================================================
# MODEL 2: LOGISTIC REGRESSION (L2 / Ridge)
# =============================================================================

set.seed(42)
cv_logit <- cv.glmnet(as.matrix(X_train_scaled), y_train,
                      family       = "binomial",
                      alpha        = 0,
                      nfolds       = 5,
                      type.measure = "class")

pred_logit <- predict(cv_logit,
                      newx = as.matrix(X_test_scaled),
                      s    = "lambda.min",
                      type = "class")
pred_logit <- as.factor(as.vector(pred_logit))

results_list[["Logistic Regression"]] <- get_metrics(pred_logit, y_test, "Logistic Regression")

# =============================================================================
# MODEL 3: RANDOM FOREST
# =============================================================================

set.seed(42)
rf_model <- randomForest(x          = X_train,
                         y          = y_train,
                         ntree      = 500,
                         mtry       = floor(sqrt(ncol(X_train))),
                         importance = TRUE)

pred_rf <- predict(rf_model, newdata = X_test)
results_list[["Random Forest"]] <- get_metrics(pred_rf, y_test, "Random Forest")

# Table 10: Top 10 Feature Importances
imp_df <- data.frame(
  Feature    = rownames(importance(rf_model)),
  MeanDecAcc = round(importance(rf_model)[, "MeanDecreaseAccuracy"], 4)
) %>%
  arrange(desc(MeanDecAcc)) %>%
  head(10)

kable(imp_df, caption = "Table 10: Random Forest Top 10 Features by Mean Decrease in Accuracy",
      row.names = FALSE)

# =============================================================================
# MODEL 4: SOFT VOTING ENSEMBLE
# =============================================================================

prob_logit <- predict(cv_logit,
                      newx = as.matrix(X_test_scaled),
                      s    = "lambda.min",
                      type = "response")
prob_logit <- as.numeric(prob_logit)

prob_rf       <- predict(rf_model, newdata = X_test, type = "prob")[, "1"]
prob_ensemble <- (prob_logit + prob_rf) / 2
pred_ensemble <- as.factor(ifelse(prob_ensemble >= 0.5, 1, 0))

results_list[["Soft Voting Ensemble"]] <- get_metrics(pred_ensemble, y_test, "Soft Voting Ensemble")

# =============================================================================
# RESULTS
# =============================================================================

# Table 11: All Model Performance
results_df <- do.call(rbind, results_list)
rownames(results_df) <- NULL

kable(results_df,
      caption = "Table 11: All Model Performance on Held-Out Test Set",
      align   = c("l", "c", "c", "c", "c"))

# Figure 12: Model Comparison Plot
results_long <- results_df %>%
  tidyr::pivot_longer(cols = c(Accuracy, Precision, Recall, F1),
                      names_to = "Metric", values_to = "Value")

ggplot(results_long, aes(x = reorder(Model, Value), y = Value, fill = Metric)) +
  geom_col(position = "dodge", alpha = 0.85) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray40", linewidth = 0.7) +
  scale_fill_manual(values = c(
    "Accuracy"  = "#2980b9",
    "Precision" = "#27ae60",
    "Recall"    = "#e67e22",
    "F1"        = "#8e44ad"
  )) +
  coord_flip() +
  labs(
    title   = "Figure 12: Model Comparison Across Evaluation Metrics",
    x       = "Model",
    y       = "Score",
    fill    = "Metric",
    caption = "Note: Dashed line at 0.5 represents the random-chance baseline."
  ) +
  theme_minimal() +
  theme(
    legend.position       = "bottom",
    plot.caption          = element_text(hjust = 0, size = 9, color = "gray30"),
    plot.caption.position = "plot"
  )
