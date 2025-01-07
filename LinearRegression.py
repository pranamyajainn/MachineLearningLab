# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset from the pendrive
file_path = "E:/Dataset/BostonHousing.csv"  # Replace with the actual file path
df = pd.read_csv(file_path)  # Load the CSV file into a DataFrame
#or
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)

# Step 3: Quick exploration
print("First 5 rows of the dataset:\n", df.head())  # Display first 5 rows
print("\nCheck for missing values:\n", df.isnull().sum())  # Check for null values

# Step 4: Visualize data
# Scatterplot of a feature (e.g., number of rooms) vs price
sns.scatterplot(x=df['rm'], y=df['medv'])
plt.title("Rooms vs House Price")
plt.xlabel("Number of Rooms")
plt.ylabel("Price")
plt.show()

# Correlation heatmap
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 5: Prepare data
X = df.drop('medv', axis=1)  # Features
y = df['medv']  # Target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 7: Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse}")
print(f"R-squared Value: {r2}")

# Step 8: Visualize predictions
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Prices")
plt.show()
