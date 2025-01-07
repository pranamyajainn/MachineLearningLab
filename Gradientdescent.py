# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load Dataset
california = fetch_california_housing()
df = pd.DataFrame(california.data, columns=california.feature_names)
df['PRICE'] = california.target

# Step 3: Display First 5 Rows
print("First 5 rows of the dataset:\n", df.head())

# Step 4: Check for Missing Values
print("\nChecking for missing values:\n", df.isnull().sum())

# Step 5: Visualize the Data
# Scatterplot for a key feature (e.g., Median Income) vs Price
sns.scatterplot(x=df['MedInc'], y=df['PRICE'])
plt.title("Median Income vs Price")
plt.xlabel("Median Income")
plt.ylabel("Price")
plt.show()

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 6: Covariance and Correlation
print("\nCovariance Matrix:\n", df.cov())
print("\nCorrelation Matrix:\n", df.corr())

# Step 7: Apply Gradient Descent for a Simple Linear Regression

# Prepare Data
X = df['MedInc'].values  # Using 'Median Income' as a single feature
y = df['PRICE'].values  # Target variable (price)

# Normalize the feature for better gradient descent performance
X = (X - np.mean(X)) / np.std(X)

# Add a bias column for the intercept
X = np.c_[np.ones(X.shape[0]), X]  # Add a column of ones (bias term)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize parameters
theta = np.zeros(X_train.shape[1])  # Initialize weights to zero
alpha = 0.01  # Learning rate
iterations = 1000  # Number of iterations
m = len(y_train)  # Number of training examples

# Gradient Descent Algorithm
for i in range(iterations):
    predictions = X_train.dot(theta)  # Predictions
    errors = predictions - y_train  # Errors
    gradients = (1 / m) * X_train.T.dot(errors)  # Compute gradients
    theta -= alpha * gradients  # Update weights

# Print the final model parameters
print("\nModel Parameters after Gradient Descent:", theta)

# Step 8: Predict and Evaluate
# Predict on test data
y_pred = X_test.dot(theta)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-squared Value: {r2}")

# Step 9: Visualize Predictions
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
