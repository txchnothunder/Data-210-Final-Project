library(ggplot2)
library(here)
library(tidyr)

nvda_data <- read.csv(here("data", "NVDA_yfinance_clean.csv"))
df <- nvda_data

# 3. Perform your operations
df$Date <- as.Date(df$Date)
df$Volume_M <- df$Volume / 1e6