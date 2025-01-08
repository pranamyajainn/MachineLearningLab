from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score (Accuracy): {r2:.2f}")

# Step 2: Display the first 5 rows
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df= data.frame
print("First 5 rows of the dataset:")
print(df.head())
  

covariance = df.cov()
correlation = df.corr()
print("\nCovariance matrix (showing first 5 rows):")
print(covariance.head())
print("\nCorrelation matrix (showing first 5 rows):")
print(correlation.head())

plt.scatter(df["MedInc"], df["MedHouseVal"], alpha=0.5, color='green')
plt.title("Median Income vs Median House Value")
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.show()
