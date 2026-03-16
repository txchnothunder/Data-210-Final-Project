# =============================================================================
# DATA 210 Final Project - Full Analysis Script
# NVIDIA (NVDA) Stock Prediction
# Author: Ethan Wong
#
# This script contains ALL code from the project, including models that were
# removed from the final report for conciseness (XGBoost, SVM, LSTM, Ensemble
# Stacking). The final Rmd document uses: ARIMA, Logistic Regression, Random Forest.
# =============================================================================


# -----------------------------------------------------------------------------
# 0. SETUP
# -----------------------------------------------------------------------------

required_packages <- c(
  "here", "dplyr", "tidyr", "ggplot2", "knitr",
  "tseries", "forecast", "TTR", "zoo",
  "randomForest", "xgboost", "e1071", "glmnet", "keras"
)

lapply(required_packages, function(pkg) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
})

source(here("scripts/01_read_df.R"))


# -----------------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# -----------------------------------------------------------------------------

# Price-derived features
df <- df %>%
  mutate(
    daily_range    = High - Low,
    overnight_gap  = Open - lag(Close),
    close_position = (Close - Low) / (High - Low + 1e-8),
    log_return     = log(Close / lag(Close))
  )

# Rolling and lagged features
df <- df %>%
  mutate(
    lag1 = lag(log_return, 1),
    lag2 = lag(log_return, 2),
    lag3 = lag(log_return, 3),
    lag4 = lag(log_return, 4),
    lag5 = lag(log_return, 5),

    vol_5d  = rollapply(log_return, width = 5,  FUN = sd,   fill = NA, align = "right"),
    vol_20d = rollapply(log_return, width = 20, FUN = sd,   fill = NA, align = "right"),
    ma_5d   = rollapply(log_return, width = 5,  FUN = mean, fill = NA, align = "right"),
    ma_20d  = rollapply(log_return, width = 20, FUN = mean, fill = NA, align = "right"),

    sma_20         = SMA(Close, n = 20),
    price_vs_sma20 = Close / sma_20 - 1
  )

# Technical indicators
df$rsi_14      <- RSI(df$Close, n = 14)
macd_vals      <- MACD(df$Close, nFast = 12, nSlow = 26, nSig = 9)
df$macd        <- macd_vals[, "macd"]
df$macd_signal <- macd_vals[, "signal"]
df$macd_hist   <- df$macd - df$macd_signal

bb          <- BBands(df$Close, n = 20, sd = 2)
df$bb_pct_b <- bb[, "pctB"]
df$bb_width <- (bb[, "up"] - bb[, "dn"]) / bb[, "mavg"]

# Volume features
df <- df %>%
  mutate(
    vol_ma20   = rollapply(Volume, width = 20, FUN = mean, fill = NA, align = "right"),
    rel_volume = Volume / vol_ma20,
    log_volume = log(Volume + 1)
  )

# Binary target: will next day close higher?
df <- df %>%
  mutate(target = as.integer(lead(Close) > Close))

df_model <- df %>% drop_na()
cat("Rows after feature engineering:", nrow(df_model), "\n")


# -----------------------------------------------------------------------------
# 2. EDA PLOTS
# -----------------------------------------------------------------------------

# Figure 1: Log-transformed price distributions
par(mfrow = c(2, 2), oma = c(0, 0, 3, 0), mar = c(4, 4, 2, 1))
hist(log(df$Close), main = "Log(Close Price)", xlab = "log(USD)", col = "#76b900", breaks = 60)
hist(log(df$High),  main = "Log(High Price)",  xlab = "log(USD)", col = "#4a90d9", breaks = 60)
hist(log(df$Low),   main = "Log(Low Price)",   xlab = "log(USD)", col = "#e05c5c", breaks = 60)
hist(log(df$Open),  main = "Log(Open Price)",  xlab = "log(USD)", col = "#f4a83a", breaks = 60)
mtext("Figure 1: Log-Transformed Price Distributions", side = 3, outer = TRUE, line = 1, cex = 1.2, font = 2)
par(mfrow = c(1, 1), oma = c(0, 0, 0, 0), mar = c(5, 4, 4, 2))

# Figure 2: Log return distribution
hist(df_model$log_return,
     main   = "Figure 2: Distribution of Daily Log Returns",
     xlab   = "Log Return",
     col    = "#2ecc71",
     breaks = 80)

# Figure 3: Close price over time
ggplot(df, aes(x = Date, y = Close)) +
  geom_line(color = "#76b900", linewidth = 0.6) +
  labs(title = "Figure 3: NVDA Close Price (2016 to 2025)", x = "Date", y = "Close (USD)") +
  theme_minimal()

# Figure 4: Volume over time
ggplot(df, aes(x = Date, y = Volume_M)) +
  geom_line(color = "#9b59b6", linewidth = 0.3, alpha = 0.3) +
  geom_smooth(method = "loess", span = 0.15, se = TRUE,
              color = "#6c3483", fill = "#d2b4de", linewidth = 1.1) +
  labs(title = "Figure 4: NVDA Daily Trading Volume (Millions of Shares)",
       x = "Date", y = "Volume (Millions)") +
  theme_minimal()

# Figure 5: Log returns over time with volatility envelope
df_model <- df_model %>%
  mutate(smooth_vol_upper =  2 * vol_20d,
         smooth_vol_lower = -2 * vol_20d)

ggplot(df_model, aes(x = Date)) +
  geom_ribbon(aes(ymin = smooth_vol_lower, ymax = smooth_vol_upper,
                  fill = "2x 20-Day Vol Envelope"), alpha = 0.35) +
  geom_line(aes(y = log_return, color = "Raw Log Return"), linewidth = 0.25, alpha = 0.4) +
  geom_smooth(aes(y = log_return, color = "LOESS Trend"),
              method = "loess", span = 0.1, se = FALSE, linewidth = 1.0) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_manual(name = "Series",
                     values = c("Raw Log Return" = "#2ecc71", "LOESS Trend" = "#1a7a40")) +
  scale_fill_manual(name = "Envelope",
                    values = c("2x 20-Day Vol Envelope" = "#a9dfbf")) +
  labs(title = "Figure 5: NVDA Daily Log Returns Over Time", x = "Date", y = "Log Return") +
  theme_minimal()

# Figure 6: Rolling volatility
df_model %>%
  tidyr::pivot_longer(cols = c(vol_5d, vol_20d), names_to = "Window", values_to = "Volatility") %>%
  mutate(Window = recode(Window, vol_5d = "5-Day", vol_20d = "20-Day")) %>%
  ggplot(aes(x = Date, y = Volatility, color = Window, alpha = Window, linewidth = Window)) +
  geom_line() +
  scale_color_manual(values = c("5-Day" = "#e67e22", "20-Day" = "#c0392b")) +
  scale_alpha_manual(values = c("5-Day" = 0.35, "20-Day" = 0.9)) +
  scale_linewidth_manual(values = c("5-Day" = 0.3, "20-Day" = 0.8)) +
  labs(title = "Figure 6: Rolling Volatility Over Time", x = "Date", y = "Std Dev of Log Return") +
  guides(alpha = "none", linewidth = "none") +
  theme_minimal()

# Figure 7: RSI over time
ggplot(df_model, aes(x = Date, y = rsi_14)) +
  annotate("rect", xmin = min(df_model$Date), xmax = max(df_model$Date),
           ymin = 70, ymax = 100, fill = "#fadbd8", alpha = 0.4) +
  annotate("rect", xmin = min(df_model$Date), xmax = max(df_model$Date),
           ymin = 0,  ymax = 30,  fill = "#d6eaf8", alpha = 0.4) +
  geom_line(aes(color = "Raw RSI"), linewidth = 0.25, alpha = 0.4) +
  geom_smooth(aes(color = "LOESS Trend"), method = "loess", span = 0.08,
              se = FALSE, linewidth = 1.0) +
  geom_hline(yintercept = c(70, 30), linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
  scale_color_manual(values = c("Raw RSI" = "#3498db", "LOESS Trend" = "#1a5276")) +
  labs(title = "Figure 7: 14-Day RSI Over Time", x = "Date", y = "RSI") +
  theme_minimal()

# Figure 8: MACD histogram
ggplot(df_model, aes(x = Date, y = macd_hist, fill = macd_hist > 0)) +
  geom_col(width = 1, alpha = 0.75) +
  scale_fill_manual(values = c("TRUE" = "#27ae60", "FALSE" = "#e74c3c"),
                    labels = c("TRUE" = "Bullish", "FALSE" = "Bearish"),
                    name = "Market Momentum") +
  geom_smooth(aes(x = Date, y = macd_hist), method = "loess", span = 0.1,
              se = FALSE, color = "#2c3e50", linewidth = 0.9, inherit.aes = FALSE) +
  geom_hline(yintercept = 0, color = "black", linewidth = 0.5) +
  labs(title = "Figure 8: MACD Histogram Over Time", x = "Date", y = "MACD minus Signal") +
  theme_minimal()


# -----------------------------------------------------------------------------
# 3. STATIONARITY TESTS
# -----------------------------------------------------------------------------

cat("=== ADF Test: Raw Close Price ===\n")
print(adf.test(df$Close))

log_ret_clean <- na.omit(df$log_return)
cat("\n=== ADF Test: Log Returns ===\n")
print(adf.test(log_ret_clean))


# -----------------------------------------------------------------------------
# 4. ACF / PACF
# -----------------------------------------------------------------------------

par(mar = c(5, 4, 4, 2))
Acf(log_ret_clean,  main = "Figure 9: ACF of NVDA Log Returns",  lag.max = 35)
Pacf(log_ret_clean, main = "Figure 10: PACF of NVDA Log Returns", lag.max = 35)


# -----------------------------------------------------------------------------
# 5. CLUSTERING (EDA requirement -- not used in modeling)
# -----------------------------------------------------------------------------

features        <- df %>% select(Close, High, Low, Open, Volume)
scaled_features <- scale(features)
set.seed(123)
kmeans_result <- kmeans(scaled_features, centers = 3)
df$cluster    <- as.factor(kmeans_result$cluster)

ggplot(df, aes(x = Close, y = Volume, color = cluster)) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Figure 10: K-Means Clustering of NVDA Price and Volume",
       x = "Close Price", y = "Volume", color = "Cluster ID")


# -----------------------------------------------------------------------------
# 6. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# 7. EVALUATION HELPER
# -----------------------------------------------------------------------------

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
  f1  <- if (!is.na(pre) && !is.na(rec) && (pre + rec) > 0) 2 * pre * rec / (pre + rec) else NA
  data.frame(Model = model_name, Accuracy = round(acc, 4), Precision = round(pre, 4),
             Recall = round(rec, 4), F1 = round(f1, 4), row.names = NULL)
}

results_list <- list()


# -----------------------------------------------------------------------------
# 8. MODEL 1: ARIMA (used in final report)
# -----------------------------------------------------------------------------

train_returns <- train_df$log_return
test_returns  <- test_df$log_return

set.seed(42)
arima_model <- auto.arima(train_returns, seasonal = FALSE, stepwise = TRUE,
                          approximation = FALSE)
print(summary(arima_model))
checkresiduals(arima_model)

# Optional: forecast plot (removed from final report to reduce figure count)
arima_order <- arimaorder(arima_model)
p_a <- arima_order[1]; d_a <- arima_order[2]; q_a <- arima_order[3]

all_returns   <- c(train_returns, test_returns)
n_test        <- length(test_returns)
rolling_preds <- numeric(n_test)
rolling_lower <- numeric(n_test)
rolling_upper <- numeric(n_test)

for (i in seq_len(n_test)) {
  history          <- all_returns[1:(n_train + i - 1)]
  fit_i            <- Arima(history, order = c(p_a, d_a, q_a))
  fc_i             <- forecast(fit_i, h = 1, level = 95)
  rolling_preds[i] <- as.numeric(fc_i$mean)
  rolling_lower[i] <- as.numeric(fc_i$lower)
  rolling_upper[i] <- as.numeric(fc_i$upper)
}

forecast_df <- data.frame(
  Date   = test_df$Date,
  Actual = test_returns,
  Pred   = rolling_preds,
  Lower  = rolling_lower,
  Upper  = rolling_upper
)

ggplot(forecast_df, aes(x = Date)) +
  geom_ribbon(aes(ymin = Lower, ymax = Upper), fill = "#aed6f1", alpha = 0.5) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.4, alpha = 0.7) +
  geom_line(aes(y = Pred,   color = "ARIMA Forecast"), linewidth = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray40") +
  scale_color_manual(values = c("Actual" = "#2c3e50", "ARIMA Forecast" = "#2980b9")) +
  labs(title = "ARIMA Rolling One-Step-Ahead Forecast vs Actual Log Returns",
       x = "Date", y = "Log Return", color = NULL,
       caption = "Note: Blue ribbon is the 95% forecast interval.") +
  theme_minimal()

# Classify by sign of forecast
pred_arima <- as.factor(ifelse(rolling_preds >= 0, 1, 0))
results_list[["ARIMA"]] <- get_metrics(pred_arima, y_test, "ARIMA")


# -----------------------------------------------------------------------------
# 9. BASELINE: LOGISTIC REGRESSION (used in final report)
# -----------------------------------------------------------------------------

set.seed(42)
cv_logit <- cv.glmnet(as.matrix(X_train_scaled), y_train,
                      family = "binomial", alpha = 0, nfolds = 5,
                      type.measure = "class")

pred_logit <- as.factor(as.vector(predict(cv_logit,
                                          newx = as.matrix(X_test_scaled),
                                          s    = "lambda.min",
                                          type = "class")))

results_list[["Logistic Regression"]] <- get_metrics(pred_logit, y_test, "Logistic Regression")


# -----------------------------------------------------------------------------
# 10. MODEL 2: RANDOM FOREST (used in final report)
# -----------------------------------------------------------------------------

set.seed(42)
rf_model <- randomForest(x = X_train, y = y_train, ntree = 500,
                         mtry = floor(sqrt(ncol(X_train))), importance = TRUE)

pred_rf <- predict(rf_model, newdata = X_test)
results_list[["Random Forest"]] <- get_metrics(pred_rf, y_test, "Random Forest")

# Feature importance plot (removed from final report, kept here for reference)
imp_df <- data.frame(
  Feature    = rownames(importance(rf_model)),
  MeanDecAcc = importance(rf_model)[, "MeanDecreaseAccuracy"]
) %>%
  arrange(desc(MeanDecAcc)) %>%
  head(15)

ggplot(imp_df, aes(x = reorder(Feature, MeanDecAcc), y = MeanDecAcc)) +
  geom_col(fill = "#27ae60", alpha = 0.85) +
  coord_flip() +
  labs(title = "Random Forest Feature Importance (Top 15)",
       x = "Feature", y = "Mean Decrease in Accuracy") +
  theme_minimal()


# -----------------------------------------------------------------------------
# 13. SOFT VOTING ENSEMBLE (used in final report)
# -----------------------------------------------------------------------------

# Logistic Regression probabilities
prob_logit <- as.numeric(predict(cv_logit,
                                 newx = as.matrix(X_test_scaled),
                                 s    = "lambda.min",
                                 type = "response"))

# Random Forest probabilities
prob_rf <- predict(rf_model, newdata = X_test, type = "prob")[, "1"]

# Average (soft vote)
prob_ensemble <- (prob_logit + prob_rf) / 2
pred_ensemble <- as.factor(ifelse(prob_ensemble >= 0.5, 1, 0))
results_list[["Soft Voting Ensemble"]] <- get_metrics(pred_ensemble, y_test, "Soft Voting Ensemble")


# -----------------------------------------------------------------------------
# 14. XGBOOST (removed from final report)
# -----------------------------------------------------------------------------

y_train_num <- as.numeric(as.character(y_train))
y_test_num  <- as.numeric(as.character(y_test))

dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train_num)
dtest  <- xgb.DMatrix(data = as.matrix(X_test),  label = y_test_num)

set.seed(42)
xgb_model <- xgb.train(
  params  = list(objective = "binary:logistic", eval_metric = "logloss",
                 eta = 0.05, max_depth = 4, subsample = 0.8,
                 colsample_bytree = 0.8, lambda = 1, gamma = 0),
  data    = dtrain,
  nrounds = 500,
  verbose = 0
)

pred_xgb <- as.factor(ifelse(predict(xgb_model, dtest) >= 0.5, 1, 0))
results_list[["XGBoost"]] <- get_metrics(pred_xgb, y_test, "XGBoost")


# -----------------------------------------------------------------------------
# 12. SVM (removed from final report)
# -----------------------------------------------------------------------------

gamma_val <- 1 / (ncol(X_train_scaled) * mean(apply(X_train_scaled, 2, var)))

set.seed(42)
svm_model <- svm(x = X_train_scaled, y = y_train,
                 kernel = "radial", cost = 1, gamma = gamma_val, probability = TRUE)

pred_svm <- predict(svm_model, newdata = X_test_scaled)
results_list[["SVM (RBF)"]] <- get_metrics(pred_svm, y_test, "SVM (RBF)")


# -----------------------------------------------------------------------------
# 13. LSTM (removed from final report -- requires keras)
# -----------------------------------------------------------------------------

build_sequences <- function(X, y, lookback = 20) {
  n     <- nrow(X)
  idx   <- (lookback + 1):n
  X_seq <- array(NA, dim = c(length(idx), lookback, ncol(X)))
  for (i in seq_along(idx)) {
    X_seq[i, , ] <- as.matrix(X[(idx[i] - lookback):(idx[i] - 1), ])
  }
  list(X = X_seq, y = y[idx])
}

train_seq <- build_sequences(as.data.frame(X_train_scaled),
                             as.numeric(as.character(y_train)), lookback = 20)
test_seq  <- build_sequences(as.data.frame(X_test_scaled),
                             as.numeric(as.character(y_test)),  lookback = 20)

input_layer <- layer_input(shape = c(20, 20))
lstm_model  <- keras_model(
  inputs  = input_layer,
  outputs = input_layer %>%
    layer_lstm(units = 64, return_sequences = FALSE) %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dropout(rate = 0.2) %>%
    layer_dense(units = 1, activation = "sigmoid")
)

lstm_model %>% compile(optimizer = optimizer_adam(learning_rate = 0.001),
                       loss = "binary_crossentropy", metrics = list("accuracy"))

early_stop <- callback_early_stopping(monitor = "val_loss", patience = 10,
                                      restore_best_weights = TRUE)

set.seed(42)
lstm_model %>% fit(x = train_seq$X, y = train_seq$y,
                   epochs = 100, batch_size = 32, validation_split = 0.1,
                   callbacks = list(early_stop), verbose = 0)

pred_lstm   <- as.factor(ifelse(predict(lstm_model, test_seq$X) >= 0.5, 1, 0))
y_test_lstm <- as.factor(test_seq$y)
results_list[["LSTM"]] <- get_metrics(pred_lstm, y_test_lstm, "LSTM")


# -----------------------------------------------------------------------------
# 14. ENSEMBLE STACKING (removed from final report)
# -----------------------------------------------------------------------------

set.seed(42)
k_folds  <- 5
fold_ids <- cut(seq_len(nrow(X_train)), breaks = k_folds, labels = FALSE)

oof_rf  <- numeric(nrow(X_train))
oof_xgb <- numeric(nrow(X_train))
oof_svm <- numeric(nrow(X_train))

for (fold in 1:k_folds) {
  val_idx   <- which(fold_ids == fold)
  train_idx <- which(fold_ids != fold)

  Xtr  <- X_train[train_idx, ]; ytr  <- y_train[train_idx]
  Xval <- X_train[val_idx, ];   yval <- y_train[val_idx]
  Xtr_s  <- scale(Xtr)
  Xval_s <- scale(Xval, center = attr(Xtr_s, "scaled:center"),
                        scale  = attr(Xtr_s, "scaled:scale"))

  rf_fold <- randomForest(x = Xtr, y = ytr, ntree = 300, mtry = floor(sqrt(ncol(Xtr))))
  oof_rf[val_idx] <- predict(rf_fold, newdata = Xval, type = "prob")[, "1"]

  dtr  <- xgb.DMatrix(as.matrix(Xtr),  label = as.numeric(as.character(ytr)))
  dval <- xgb.DMatrix(as.matrix(Xval), label = as.numeric(as.character(yval)))
  xgb_fold <- xgb.train(
    params  = list(objective = "binary:logistic", eval_metric = "logloss",
                   eta = 0.05, max_depth = 4, subsample = 0.8, colsample_bytree = 0.8),
    data = dtr, nrounds = 300, verbose = 0
  )
  oof_xgb[val_idx] <- predict(xgb_fold, dval)

  gamma_fold <- 1 / (ncol(Xtr_s) * mean(apply(Xtr_s, 2, var)))
  svm_fold   <- svm(x = Xtr_s, y = ytr, kernel = "radial",
                    cost = 1, gamma = gamma_fold, probability = TRUE)
  svm_probs        <- attr(predict(svm_fold, Xval_s, probability = TRUE), "probabilities")
  oof_svm[val_idx] <- svm_probs[, "1"]
}

meta_train <- data.frame(rf = oof_rf, xgb = oof_xgb, svm = oof_svm)
meta_y     <- as.numeric(as.character(y_train))
meta_model <- cv.glmnet(as.matrix(meta_train), meta_y,
                        family = "binomial", alpha = 0, nfolds = 5)

rf_test_prob  <- predict(rf_model, newdata = X_test, type = "prob")[, "1"]
xgb_test_prob <- predict(xgb_model, dtest)
svm_test_prob <- attr(predict(svm_model, X_test_scaled, probability = TRUE),
                      "probabilities")[, "1"]

meta_test       <- data.frame(rf = rf_test_prob, xgb = xgb_test_prob, svm = svm_test_prob)
pred_stack_prob <- predict(meta_model, newx = as.matrix(meta_test),
                           s = "lambda.min", type = "response")
pred_stack      <- as.factor(ifelse(pred_stack_prob >= 0.5, 1, 0))
results_list[["Ensemble Stack"]] <- get_metrics(pred_stack, y_test, "Ensemble Stack")


# -----------------------------------------------------------------------------
# 15. RESULTS SUMMARY (all models)
# -----------------------------------------------------------------------------

results_df <- do.call(rbind, results_list)
rownames(results_df) <- NULL
print(results_df)

results_long <- results_df %>%
  tidyr::pivot_longer(cols = c(Accuracy, Precision, Recall, F1),
                      names_to = "Metric", values_to = "Value")

ggplot(results_long, aes(x = reorder(Model, Value), y = Value, fill = Metric)) +
  geom_col(position = "dodge", alpha = 0.85) +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "gray40", linewidth = 0.7) +
  scale_fill_manual(values = c("Accuracy"  = "#2980b9", "Precision" = "#27ae60",
                               "Recall"    = "#e67e22", "F1"        = "#8e44ad")) +
  coord_flip() +
  labs(title = "All Models: Performance Comparison",
       x = "Model", y = "Score", fill = "Metric",
       caption = "Dashed line at 0.5 = random-chance baseline.") +
  theme_minimal() +
  theme(legend.position = "bottom")
