from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Dataset
data = [
    ("I love this sandwich", "pos"),
    ("This is an amazing place", "pos"),
    ("I feel very good about these cheese", "pos"),
    ("This is my best work", "pos"),
    ("What an awesome view", "pos"),
    ("I do not like this restaurant", "neg"),
    ("I am tired of this stuff", "neg"),
    ("I can’t deal with this", "neg"),
    ("He is my sworn enemy", "neg"),
    ("My boss is horrible", "neg"),
    ("This is an awesome place", "pos"),
    ("I do not like the taste of this juice", "neg"),
    ("I love to dance", "pos"),
    ("I am sick and tired of this place", "neg"),
    ("What a great holiday", "pos"),
    ("That is a bad locality to stay", "neg"),
    ("We will have good fun tomorrow", "pos"),
    ("I went to my enemys house today", "neg"),
]

# Splitting data into features (X) and labels (y)
X, y = zip(*data)

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Vectorizing the text data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)  # Ensure transformation is applied to X_test

# Training the Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Making predictions
y_pred = classifier.predict(X_test_vectorized)  # Use transformed test data here

# Evaluating the classifier
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label="pos")
recall = recall_score(y_test, y_pred, pos_label="pos")
conf_matrix = confusion_matrix(y_test, y_pred, labels=["pos", "neg"])

# Results
print("Total Instances in Dataset:", len(data))
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Confusion Matrix:")
print(conf_matrix)
