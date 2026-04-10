import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("customers.csv")

print("Dataset Loaded Successfully\n")

# Basic info
print(data.head())
print("\nShape:", data.shape)

# Select features (simple)
X = data[["Income", "Recency"]]

# Apply KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
data["Cluster"] = kmeans.fit_predict(X)

print("\nClustering Done!")

# Plot
plt.scatter(X["Income"], X["Recency"], c=data["Cluster"])
plt.xlabel("Income")
plt.ylabel("Recency")
plt.title("Customer Segmentation")
plt.savefig("segmentation.png")
plt.show()