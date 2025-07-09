
import pandas as pd

# Load the dataset
df = pd.read_excel('online_retail_II.xlsx')

# Display basic information
print('Dataset Head:')
print(df.head())

print('\nDataset Info:')
df.info()

print('\nDataset Description:')
print(df.describe())

print('\nMissing Values:')
print(df.isnull().sum())


