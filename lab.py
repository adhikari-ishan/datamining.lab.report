
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