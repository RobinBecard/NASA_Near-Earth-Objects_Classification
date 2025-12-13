from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.models.base_classifier import BaseModel
from src.config import get_config
import numpy as np
import matplotlib.pyplot as plt

class KMeansClusterer(BaseModel):
    """
    Specific implementation for K-Means (Clustering).
    Inherits from BaseModel but adapts methods as it is unsupervised.
    """

    def _build_model(self):
        """
        Builds the KMeans model with hyperparameters from config.yaml.
        
        Key Hyperparameters:
            - n_clusters: The number of groups (K) to find.
            - init: Initialization method ('k-means++' is recommended).
            - n_init: Number of times the algo will be run with different seeds.
        """
        config = get_config()
        default_params = config.get_param('models.kmeans', {})
        
        self.model = KMeans(
            n_clusters=self.params.get('n_clusters', default_params.get('n_clusters', 3)),
            init=self.params.get('init', default_params.get('init', 'k-means++')),
            n_init=self.params.get('n_init', default_params.get('n_init', 10)),
            max_iter=self.params.get('max_iter', default_params.get('max_iter', 300)),
            random_state=self.params.get('random_state', default_params.get('random_state', 42))
        )

    def train(self, X_train, y_train=None):
        """
        Override of train method.
        K-Means is unsupervised, so it ignores y_train.
        """
        print(f"--> Training {self.name} (Unsupervised)...")
        if self.model is None:
            self._build_model()
        
        self.model.fit(X_train)

    def evaluate_clustering(self, X):
        """
        Specific evaluation for clustering (without ground truth labels).
        Displays Inertia and Silhouette Score.
        """
        if self.model is None:
            return
        
        labels = self.model.predict(X)
        
        # Inertia: Sum of squared distances of samples to their closest cluster center.
        # Lower is better (but beware of overfitting if k is large).
        inertia = self.model.inertia_
        
        # Silhouette Score: Measures if objects are well within their cluster and far from others.
        # Ranges from -1 (bad) to +1 (excellent).
        try:
            sil_score = silhouette_score(X, labels)
        except:
            sil_score = "N/A (Need >1 cluster)"

        print(f"\nCLUSTERING RESULTS - {self.name}:")
        print("  ─" * 30)
        print(f"  - Inertia (Distortion): {inertia:.4f}")
        print(f"  - Silhouette Score:     {sil_score if isinstance(sil_score, str) else f'{sil_score:.4f}'}")
        
        return {"inertia": inertia, "silhouette": sil_score}

    def plot_clusters(self, X, title="K-Means Clustering"):
        """
        Visualizes clusters, Voronoi boundaries, and CENTROIDS.
        Requires 2D data.
        """
        if self.model is None:
            print("Error: Model not trained.")
            return

        X = np.array(X)
        if X.shape[1] != 2:
            print(f"Visualization impossible in {X.shape[1]}D. Use 2 features.")
            return

        h = 0.02
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        Z = self.model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        
        # Colored background by cluster region
        plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap='Pastel1',
                   aspect='auto', origin='lower')

        y_pred = self.model.predict(X)
        plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='Set1', edgecolors='k', alpha=0.7)

        centroids = self.model.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1],
                    marker='x', s=169, linewidths=3,
                    color='w', zorder=10, label='Centroids')

        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.legend()
        plt.grid(False)
        plt.show()