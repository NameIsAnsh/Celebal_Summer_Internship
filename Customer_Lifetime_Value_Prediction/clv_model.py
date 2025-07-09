
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.utils import calibration_and_holdout_data, summary_data_from_transaction_data

# Load the processed RFM data
rfm_df = pd.read_csv("rfm_data.csv", index_col="Customer ID")

# Filter out customers with Frequency <= 0 for Gamma-Gamma model
rfm_df_filtered = rfm_df[rfm_df["Frequency"] > 0]

# BG/NBD Model (BetaGeoFitter)
bgf = BetaGeoFitter(penalizer_coef=0.1)
bgf.fit(rfm_df_filtered["Frequency"], rfm_df_filtered["Recency"], rfm_df_filtered["T"])

print("BG/NBD Model Parameters:")
print(bgf.summary)

# Gamma-Gamma Model
ggf = GammaGammaFitter(penalizer_coef=0.1)
ggf.fit(rfm_df_filtered["Frequency"], rfm_df_filtered["Monetary"])

print("\nGamma-Gamma Model Parameters:")
print(ggf.summary)

# Predict future purchases (e.g., for the next 30 days)
rfm_df_filtered["predicted_purchases"] = bgf.predict(30, rfm_df_filtered["Frequency"], rfm_df_filtered["Recency"], rfm_df_filtered["T"])

# Calculate Conditional Expected Average Profit
rfm_df_filtered["predicted_clv"] = ggf.conditional_expected_average_profit(rfm_df_filtered["Frequency"], rfm_df_filtered["Monetary"])

# Calculate CLV
# Assuming a discount rate of 0.1 and a period of 12 months for CLV calculation
# Note: This is a simplified CLV calculation. For a more robust CLV, consider customer lifetime in months/years.
rfm_df_filtered["CLV"] = ggf.customer_lifetime_value(
    bgf,
    rfm_df_filtered["Frequency"],
    rfm_df_filtered["Recency"],
    rfm_df_filtered["T"],
    rfm_df_filtered["Monetary"],
    time=12,  # months
    discount_rate=0.01 # monthly discount rate
)

print("\nRFM with Predicted Purchases and CLV Head:")
print(rfm_df_filtered.head())

# Save the results
rfm_df_filtered.to_csv("clv_predictions.csv")


