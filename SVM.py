from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Step 1: Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)  # Features as DataFrame
y = pd.Series(iris.target, name="species")  # Target as Series



# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Create an SVM classifier
svm_model = SVC(kernel='linear', random_state=42)

# Step 4: Train the classifier on the training data
svm_model.fit(X_train, y_train)

# Step 5: Evaluate the classifier on the test data  
y_pred = svm_model.predict(X_test)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the SVM model: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
