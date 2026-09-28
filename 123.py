import numpy as np 
import pandas as pd


print(pd.__version__)

#Q1
df = pd.read_csv('car_fuel_efficiency_2026.csv')
print('Q1', df)

#Q2
print('Q2', len(df))

# Q3
print('Q3', df["fuel_type"].nunique())

# Q4
print('Q4', df.isna().any().sum())

# Q5
asia_max = df.loc[
    df["origin"] == "Asia",
    "fuel_efficiency_mpg"
].max()

print('Q5', asia_max)

# Q6
median_before = df["horsepower"].median()
mode = df["horsepower"].mode().iloc[0]

median_after = df["horsepower"].fillna(mode).median()

print('Median before', median_before)
print('Mode, mode')
print('Median after', median_after)

# Q7
X = (
    df.loc[
        df["origin"] == "Asia",
        ["vehicle_weight", "model_year"]
    ]
    .head(7)
    .to_numpy()
)

XTX = X.T @ X
XTX_inv = np.linalg.inv(XTX)

y = np.array([1100, 1300, 800, 900, 1000, 1100, 1200])

w = XTX_inv @ X.T @ y

print('w', w)
print('Q7', w.sum())