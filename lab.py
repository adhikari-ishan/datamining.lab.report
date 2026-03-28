
import pandas as pd
import numpy as np

print("loading and printing data\n")

df = pd.read_csv("data.csv")
print(df.head())
print("\n")

print("Printing first 10 data\n")
print(df.head(10))
print("Printing datatypes and info\n")
print(df.dtypes)
print("\n")
print(df.info(10))
print("Calculating the missing value \n \n")
print(df.isnull().sum())

print("\n")
print(df.columns[df.isnull().any()])
print("Filling the missing values:\n")


# Fill missing values
# df['Age'].fillna(df['Age'].mean(), inplace=True)
# df['Unit_Price'].fillna(df['Unit_Price'].median(), inplace=True)
# df['Quantity'].fillna(df['Quantity'].median(), inplace=True)
# df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
# df['City'].fillna("Unknown", inplace=True)

df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Unit_Price'] = df['Unit_Price'].fillna(df['Unit_Price'].median())
df['Quantity'] = df['Quantity'].fillna(df['Quantity'].median())
df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
df['City'] = df['City'].fillna("Unknown")

print(df.isnull().sum())
print("\n")
print("Removing duplicate data \n")

# Remove duplicates
# df.drop_duplicates(inplace=True)
# print(df.shape)

# Check for duplicate rows
duplicates = df.duplicated()
# print("Number of duplicate rows:", duplicates.sum())
print(df[duplicates])
df.drop_duplicates(inplace=True)
print("After removal, number of duplicate rows:", df.duplicated().sum())
print()
print("Ensure all numerical columns are in correct format\n")

# Check current data types of all columns
numeric_columns = ['Age', 'Unit_Price', 'Quantity']  

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')  

print(df.dtypes)

print()
print("Standardize categorical values:\n")
# Standardize categorical values
df['Gender'] = df['Gender'].str.lower()
df['Product_Category'] = df['Product_Category'].str.lower()
df['City'] = df['City'].str.lower()
print(df.head(10))

# calculate mean
print()
# Example for Gender and City columns
categorical_columns = ['Gender', 'City']  # replace with your categorical columns

for col in categorical_columns:
    df[col] = df[col].str.strip().str.lower()  # remove spaces and lowercase

# Optional: check unique values
for col in categorical_columns:
    print(f"{col} unique values:", df[col].unique())
numerical_columns = ['Age', 'Unit_Price', 'Total_Amount']
print()
print("Calculating mean\n")
for col in numerical_columns:
    mean_val = df[col].mean()
    median_val = df[col].median()
    mode_val = df[col].mode()[0]
    print(f"{col}: Mean = {mean_val}, Median = {median_val}, Mode = {mode_val}")

    # Compare mean and median for skewness
    if mean_val > median_val:
        skewness = "positively skewed (right-skewed)"
    elif mean_val < median_val:
        skewness = "negatively skewed (left-skewed)"
    else:
        skewness = "approximately symmetric"

    print(f"  Skewness: {skewness}\n")

print()
print("IQR calculations\n")
    # Choose the column you want to analyze, for example 'Age'
col = 'Age'

# Calculate Q1 (25th percentile) and Q3 (75th percentile)
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate lower bound for outliers
lower_bound = Q1 - 1.5 * IQR

# Print results
print(f"{col} Q1 (25th percentile): {Q1}")
print(f"{col} Q3 (75th percentile): {Q3}")
print(f"{col} IQR: {IQR}")
print(f"{col} Lower bound (Q1 - 1.5*IQR): {lower_bound}")
upper_bound = Q3 + 1.5 * IQR


print("\n")
print("Calculate Q1, Q3, IQR, and Upper Bound\n")
# Column to analyze
col = 'Total_Amount'

# Calculate quartiles
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)

# Calculate IQR
IQR = Q3 - Q1

# Calculate lower and upper bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"{col} Q1: {Q1}")
print(f"{col} Q3: {Q3}")
print(f"{col} IQR: {IQR}")
print(f"{col} Lower bound: {lower_bound}")
print(f"{col} Upper bound: {upper_bound}")
print()
print()
print("Outlier lab\n")
# Find outlier rows
outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
print("Outliers in Total_Amount:\n", outliers)

df_no_outliers = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
df[col] = df[col].apply(lambda x: upper_bound if x > upper_bound else (lower_bound if x < lower_bound else x))

removed_rows = df[~df.index.isin(df_no_outliers.index)]
print("\nRemoved outlier rows:\n", removed_rows)

capped_rows = outliers.copy()
capped_rows[col] = capped_rows[col].apply(lambda x: upper_bound if x > upper_bound else lower_bound)
print("\nCapped outlier rows:\n", capped_rows)
print()
print("Min-Max Normalization")
print()
# Columns to normalize
min_max_cols = ['Unit_Price', 'Quantity', 'Total_Amount']

for col in min_max_cols:
    df[col + '_normalized'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

# Check the first 5 rows
print(df[min_max_cols + [c + '_normalized' for c in min_max_cols]].head())
print()
print()
print("Standardization (Z-score)")
print()

# Column to standardize
df['Age_standardized'] = (df['Age'] - df['Age'].mean()) / df['Age'].std()

# Check first 5 rows
print(df[['Age', 'Age_standardized']].head())


print()
print()
print("One-Hot Encoding for Categorical Variables\n")
print()
# Columns to encode
categorical_cols = ['Gender', 'Product_Category', 'City']

# Apply one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)  # drop_first=True avoids dummy variable trap

# Check first 5 rows
print(df_encoded.head())
print()
print("Save clean dataset\n")
print()
# Save the cleaned and encoded dataset to CSV
df_encoded.to_csv('cleaned_data.csv', index=False)
print("Cleaned dataset saved as cleaned_data.csv")
print(df.head(10))