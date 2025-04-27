import numpy as np
from typing import List, Tuple
from collections import defaultdict

class KMeans:
    def __init__(self, k: int = 3, max_iterations: int = 100, tolerance: float = 1e-4):
        """
        Initialize K-Means clustering algorithm.
        
        Args:
            k: Number of clusters
            max_iterations: Maximum number of iterations to run
            tolerance: Minimum centroid movement to continue training
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.clusters = None
        
    def initialize_centroids(self, X: np.ndarray) -> None:
        """Randomly initialize centroids by selecting k points from the dataset."""
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]
        
    def compute_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance between each point and each centroid.
        
        Returns:
            Distance matrix of shape (n_samples, k)
        """
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
    
    def assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest centroid."""
        distances = self.compute_distance(X)
        return np.argmin(distances, axis=1)
    
    def update_centroids(self, X: np.ndarray, cluster_labels: np.ndarray) -> float:
        """
        Update centroids as the mean of assigned points.
        
        Returns:
            Maximum centroid movement
        """
        new_centroids = np.zeros_like(self.centroids)
        max_movement = 0.0
        
        for i in range(self.k):
            cluster_points = X[cluster_labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
                movement = np.linalg.norm(new_centroids[i] - self.centroids[i])
                max_movement = max(max_movement, movement)
        
        self.centroids = new_centroids
        return max_movement
    
    def fit(self, X: np.ndarray) -> None:
        """
        Fit K-Means to the data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
        """
        self.initialize_centroids(X)
        
        for _ in range(self.max_iterations):
            cluster_labels = self.assign_clusters(X)
            max_movement = self.update_centroids(X, cluster_labels)
            
            if max_movement < self.tolerance:
                break
                
        self.clusters = cluster_labels
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data points."""
        if self.centroids is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        return self.assign_clusters(X)


# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    cluster1 = np.random.normal(loc=[0, 0], scale=1, size=(100, 2))
    cluster2 = np.random.normal(loc=[5, 5], scale=1, size=(100, 2))
    cluster3 = np.random.normal(loc=[-5, 5], scale=1, size=(100, 2))
    X = np.vstack([cluster1, cluster2, cluster3])
    
    # Run K-Means
    kmeans = KMeans(k=3)
    kmeans.fit(X)
    
    # Print results
    print("Centroids:")
    print(kmeans.centroids)
    
    # Visualize (requires matplotlib)
    try:
        import matplotlib.pyplot as plt
        plt.scatter(X[:, 0], X[:, 1], c=kmeans.clusters)
        plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                   marker='x', s=200, linewidths=3, color='r')
        plt.title("K-Means Clustering")
        plt.show()
    except ImportError:
        print("Matplotlib not installed. Skipping visualization.")


'''
Key Features Demonstrated:
Clean Class Structure: Well-organized methods following the scikit-learn style (fit/predict)

Type Hints: Uses Python type hints for better code clarity

Efficient Computations: Vectorized operations with NumPy

Convergence Criteria: Implements tolerance-based early stopping

Documentation: Clear docstrings explaining each method

Example Usage: Includes sample data generation and visualization

Interview Discussion Points:
Initialization Methods: This uses random initialization. You could discuss k-means++ as an alternative.

Distance Metrics: Currently uses Euclidean distance. Could discuss other options.

Convergence: The tolerance parameter controls when the algorithm stops.

Limitations:

Sensitive to initial centroids

Assumes spherical clusters

Requires specifying k

Doesn't handle categorical data well

Optimizations:

Vectorized distance computations

Early stopping

Possible parallelization

Extensions:

k-means++ initialization

Elbow method for determining k

Handling empty clusters

Mini-batch k-means for large datasets
'''
