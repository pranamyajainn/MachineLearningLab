# Step 1: Import Libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Step 2: Load the dataset from the pendrive
file_path = "E:/Datasets/iris.csv"  # Replace with the actual path to your dataset
df = pd.read_csv(file_path)

# Step 3: Display first 5 rows
print("First 5 rows of the dataset:\n", df.head())

# Step 4: Check the number of samples for each class
print("\nNumber of samples for each species:\n", df['species'].value_counts())

# Step 5: Check for missing values
print("\nChecking for missing values:\n", df.isnull().sum())

# Step 6: Visualize data
# Pairplot to observe relationships between features
sns.pairplot(df, hue='species')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Step 7: Covariance and Correlation
print("\nCovariance Matrix:\n", df.iloc[:, :-1].cov())
print("\nCorrelation Matrix:\n", df.iloc[:, :-1].corr())

# Step 8: Prepare data for Logistic Regression
X = df.iloc[:, :-1]  # Features (all columns except 'species')
y = df['species']  # Target column (species)

# Convert target to numerical values if necessary
y = y.map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Step 10: Predict and Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica']))
