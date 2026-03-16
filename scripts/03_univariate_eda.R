# Univariate Exploration of Price

par(mfrow = c(2, 2))
hist(df$Close,  main = "Close Price",  xlab = "USD", col = "#76b900", breaks = 60)
hist(df$High,   main = "High Price",   xlab = "USD", col = "#4a90d9", breaks = 60)
hist(df$Low,    main = "Low Price",    xlab = "USD", col = "#e05c5c", breaks = 60)
hist(df$Open,   main = "Open Price",   xlab = "USD", col = "#f4a83a", breaks = 60)
par(mfrow = c(1, 1))

# Log-Scale for Price
par(mfrow = c(2, 2))
hist(log(df$Close),  main = "Log(Close Price)",  xlab = "log(USD)", col = "#76b900", breaks = 60)
hist(log(df$High),   main = "Log(High Price)",   xlab = "log(USD)", col = "#4a90d9", breaks = 60)
hist(log(df$Low),    main = "Log(Low Price)",    xlab = "log(USD)", col = "#e05c5c", breaks = 60)
hist(log(df$Open),   main = "Log(Open Price)",   xlab = "log(USD)", col = "#f4a83a", breaks = 60)
par(mfrow = c(1, 1))

# Univariate Exploration of Volume

hist(df$Volume_M,
     main   = "Daily Trading Volume (Millions of Shares)",
     xlab   = "Volume (Millions)",
     col    = "#9b59b6",
     breaks = 60)

hist(log(df$Volume_M),
     main   = "Log Daily Trading Volume (Millions of Shares)",
     xlab   = "log(Volume in Millions)",
     col    = "#9b59b6",
     breaks = 50)

