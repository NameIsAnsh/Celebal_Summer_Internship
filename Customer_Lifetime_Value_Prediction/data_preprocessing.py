
import pandas as pd
import datetime as dt

# Load the dataset
df = pd.read_excel("online_retail_II.xlsx")

# Data Cleaning
# Remove rows with missing Customer ID
df.dropna(subset=["Customer ID"], inplace=True)

# Remove cancelled orders (Invoice starting with C)
df = df[~df["Invoice"].str.contains("C", na=False)]

# Remove rows with negative or zero Quantity
df = df[df["Quantity"] > 0]

# Calculate total price for each item
df["TotalPrice"] = df["Quantity"] * df["Price"]

# Convert Customer ID to string
df["Customer ID"] = df["Customer ID"].astype(str)

# Feature Engineering (RFM)
# Get the last invoice date in the dataset for recency calculation
max_invoice_date = df["InvoiceDate"].max()
analysis_date = max_invoice_date + dt.timedelta(days=1)

rfm_df = df.groupby("Customer ID").agg(
    Recency=("InvoiceDate", lambda date: (analysis_date - date.max()).days),
    Frequency=("Invoice", "nunique"),
    Monetary=("TotalPrice", "sum"),
    T=("InvoiceDate", lambda date: (analysis_date - date.min()).days) # Calculate T (customer\'s age)
)

# Filter out customers with Monetary <= 0 for Gamma-Gamma model compatibility
rfm_df = rfm_df[rfm_df["Monetary"] > 0]

print("RFM DataFrame Head:")
print(rfm_df.head())

# Save the processed data for modeling
rfm_df.to_csv("rfm_data.csv")


