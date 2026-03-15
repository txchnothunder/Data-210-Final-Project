# Bivariate Exploration of Price and Time

# Close price over time
ggplot(df, aes(x = Date, y = Close)) +
  geom_line(color = "#76b900", linewidth = 0.6) +
  labs(title = "NVDA Close Price (2016–2025)", x = "Date", y = "Close (USD)") +
  theme_minimal()

# Log scale
ggplot(df, aes(x = Date, y = Close)) +
  geom_line(color = "#76b900", linewidth = 0.6) +
  scale_y_log10() +
  labs(title = "NVDA Close Price — Log Scale", x = "Date", y = "Close (log USD)") +
  theme_minimal()

# Bivariate Exploration of Volume and Time

ggplot(df, aes(x = Date, y = Volume_M)) +
  geom_line(color = "#9b59b6", linewidth = 0.4, alpha = 0.7) +
  labs(title    = "NVDA Daily Trading Volume (Millions of Shares), 2016–2025",
       x        = "Date",
       y        = "Volume (Millions)") +
  theme_minimal()