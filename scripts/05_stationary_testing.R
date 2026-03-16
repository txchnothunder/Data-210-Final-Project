df$Date       <- as.Date(df$Date)
df$log_return <- c(NA, diff(log(df$Close)))

# ADF on raw price — expect FAIL (non-stationary)
cat("=== ADF Test: Raw Close Price ===\n")
print(adf.test(df$Close))

# ADF on log returns — expect PASS (stationary)
log_ret_clean <- na.omit(df$log_return)
cat("\n=== ADF Test: Log Returns ===\n")
print(adf.test(log_ret_clean))