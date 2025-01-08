import numpy as np
from sklearn.cluster import KMeans

# Given data points (VAR1, VAR2)
data = np.array([
    [0.1, 0.6],
    [0.15, 0.71],
    [0.08, 0.9],
    [0.16, 0.85],
    [0.2, 0.2],
    [0.25, 0.3],
    [0.24, 0.1],
    [0.3, 0.25],
    [0.85, 0.65]
])

# Initialize KMeans clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the KMeans model to the data
kmeans.fit(data)

# Predict the cluster for the given point (0.906, 0.606)
new_point = np.array([[0.906, 0.606]])
predicted_cluster = kmeans.predict(new_point)

# Output results
print("Cluster Centers:")
print(kmeans.cluster_centers_)
print(f"The point {new_point[0]} belongs to Cluster {predicted_cluster[0]}.")
