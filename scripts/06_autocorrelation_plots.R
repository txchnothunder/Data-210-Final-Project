par(mar = c(5, 4, 4, 2))

# ACF Plot
Acf(log_ret_plot, main = "Figure 5.1: ACF of NVDA Log Returns", lag.max = 35)

# PACF Plot
Pacf(log_ret_plot, main = "Figure 5.2: PACF of NVDA Log Returns", lag.max = 35)