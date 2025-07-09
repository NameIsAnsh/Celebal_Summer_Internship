
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix, plot_period_transactions
from lifetimes import BetaGeoFitter, GammaGammaFitter

# Load the processed RFM data and CLV predictions
rfm_df = pd.read_csv("rfm_data.csv", index_col="Customer ID")
clv_predictions_df = pd.read_csv("clv_predictions.csv", index_col="Customer ID")

# Filter out customers with Frequency <= 0 for Gamma-Gamma model
rfm_df_filtered = rfm_df[rfm_df["Frequency"] > 0]

# BG/NBD Model (BetaGeoFitter)
bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(rfm_df_filtered["Frequency"], rfm_df_filtered["Recency"], rfm_df_filtered["T"])

# Gamma-Gamma Model
ggf = GammaGammaFitter(penalizer_coef=0.1)
ggf.fit(rfm_df_filtered["Frequency"], rfm_df_filtered["Monetary"])

# Plot Frequency-Recency Matrix
fig = plt.figure(figsize=(10, 8))
plot_frequency_recency_matrix(bgf)
plt.title('Expected Number of Future Purchases (BG/NBD Model)')
plt.savefig('frequency_recency_matrix.png')

# Plot Probability of Being Alive
fig = plt.figure(figsize=(10, 8))
plot_probability_alive_matrix(bgf)
plt.title('Probability of Being Alive (BG/NBD Model)')
plt.savefig('probability_alive_matrix.png')

# Plot predicted vs actual for BG/NBD model (if historical data is split)
# For simplicity, we'll skip this for now as it requires splitting data into calibration and holdout.

# Visualize CLV distribution
plt.figure(figsize=(10, 6))
sns.histplot(clv_predictions_df["CLV"], bins=50, kde=True)
plt.title('Distribution of Customer Lifetime Value (CLV)')
plt.xlabel('CLV')
plt.ylabel('Number of Customers')
plt.savefig('clv_distribution.png')

print("Generated frequency_recency_matrix.png, probability_alive_matrix.png, and clv_distribution.png")


