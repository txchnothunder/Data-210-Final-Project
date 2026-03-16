# K-means Clustering
features <- df %>% select(Close, High, Low, Open, Volume)
scaled_features <- scale(features)

set.seed(123)
kmeans_result <- kmeans(scaled_features, centers = 3)

df$cluster <- as.factor(kmeans_result$cluster)

ggplot(df, aes(x = Close, y = Volume, color = cluster)) +
  geom_point(alpha = 0.5) +
  theme_minimal() +
  labs(title = "Figure 6: K-Means Clustering of NVDA Price and Volume",
       x = "Close Price",
       y = "Volume",
       color = "Cluster ID")