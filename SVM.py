from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
file_path = "E:/Dataset/Iris.csv"  # Replace with your actual file path
df = pd.read_csv(file_path)  # If file not found, it will throw FileNotFoundError

X = df.iloc[:, :-1]  # All columns except the last one as features
y = df.iloc[:, -1]   # Last column as the target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(kernel='linear', random_state=42)

# Step 5: Train the classifier on the training data
model.fit(X_train, y_train)

# Step 6: Evaluate the classifier on the test data  
y_pred = model.predict(X_test)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM model: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
