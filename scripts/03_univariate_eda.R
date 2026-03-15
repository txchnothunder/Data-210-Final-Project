# Univariate Exploration of Price (Overlapping)

ggplot(price_long, aes(x = Price, fill = PriceType, color = PriceType)) +
  geom_density(alpha = 0.25, linewidth = 0.8) +
  scale_fill_manual(values  = c(Close = "#76b900", High = "#4a90d9", 
                                Low   = "#e05c5c", Open = "#f4a83a")) +
  scale_color_manual(values = c(Close = "#76b900", High = "#4a90d9", 
                                Low   = "#e05c5c", Open = "#f4a83a")) +
  labs(title    = "NVDA — Overlapping Price Distributions (Close, High, Low, Open)",
       subtitle = "Split-adjusted daily prices, 2016–2025",
       x        = "Price (USD)",
       y        = "Density",
       fill     = "Price Type",
       color    = "Price Type") +
  theme_minimal()

# Log-Scale Exploration of Price
ggplot(price_log_long, aes(x = LogPrice, fill = PriceType, color = PriceType)) +
  geom_density(alpha = 0.25, linewidth = 0.8) +
  scale_fill_manual(values  = c(Close = "#76b900", High = "#4a90d9",
                                Low   = "#e05c5c", Open = "#f4a83a")) +
  scale_color_manual(values = c(Close = "#76b900", High = "#4a90d9",
                                Low   = "#e05c5c", Open = "#f4a83a")) +
  labs(title    = "NVDA — Overlapping Log-Price Distributions (Close, High, Low, Open)",
       subtitle = "Log-transformed split-adjusted daily prices, 2016–2025",
       x        = "log(Price) (log USD)",
       y        = "Density",
       fill     = "Price Type",
       color    = "Price Type") +
  theme_minimal()



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

boxplot(df$Volume_M,
        main = "Daily Volume Boxplot (Millions of Shares)",
        ylab = "Volume (Millions)",
        col  = "#9b59b6")

