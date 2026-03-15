price_long <- pivot_longer(df, 
                           cols = c(Close, High, Low, Open), 
                           names_to = "PriceType", 
                           values_to = "Price")

df_log <- df
df_log$Close <- log(df$Close)
df_log$High  <- log(df$High)
df_log$Low   <- log(df$Low)
df_log$Open  <- log(df$Open)

price_log_long <- pivot_longer(df_log,
                               cols      = c(Close, High, Low, Open),
                               names_to  = "PriceType",
                               values_to = "LogPrice")