
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Generate a synthetic dataset
np.random.seed(42)
n_samples = 1000
age = np.random.normal(35, 10, n_samples)
income = np.random.normal(50000, 15000, n_samples)
spending_score = np.random.normal(50, 20, n_samples)

df = pd.DataFrame({'Age': age, 'Income': income, 'Spending_Score': spending_score})

# Select the relevant features
features = ['Age', 'Income', 'Spending_Score']
df_features = df[features]

# Scale the data using StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)

# Determine the optimal number of clusters using the Elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow method, choose the optimal number of clusters
n_clusters = 5

# Perform K-means clustering
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans.fit(df_scaled)

# Get the cluster labels
labels = kmeans.labels_

# Add the cluster labels to the original dataframe
df['Cluster'] = labels

# Plot the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Spending_Score'], c=df['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.show()

# Plot the clusters with income
plt.figure(figsize=(10, 6))
plt.scatter(df['Income'], df['Spending_Score'], c=df['Cluster'], cmap='viridis')
plt.title('Customer Segmentation')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

# Analyze the clusters
print(df.groupby('Cluster').describe())