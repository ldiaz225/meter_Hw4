import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap

class DataVisualizer:
    def __init__(self, data):
        """
        data: numpy array where last column is the label
        """
        self.data = data
        self.X = data[:, :-1]
        self.y = data[:, -1]

        # Scale features immediately
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

def plot_heatmap(self):
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            pd.DataFrame(self.X_scaled).corr(),
            cmap="coolwarm",
            center=0
        )
        plt.title("Correlation Heatmap of Processed Features")
        plt.tight_layout()
        plt.show()

def plot_umap(self, n_neighbors=30, min_dist=0.1, metric="euclidean"):
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )

        embedding = umap_model.fit_transform(self.X_scaled)

        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=self.y,
            cmap="viridis",
            s=40,
            alpha=0.9
        )
        plt.colorbar(scatter, label="Label")
        plt.title("UMAP Projection of Processed Data")
        plt.xlabel("UMAP-1")
        plt.ylabel("UMAP-2")
        plt.tight_layout()
        plt.show()

'''
***How to use: 

data = preprocessor.load_data("data/Meter_A.txt")

viz = DataVisualizer(data)

viz.plot_heatmap()
viz.plot_umap()
'''