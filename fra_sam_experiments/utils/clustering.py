from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np


def compute_siluhette(embeddings, indx, ax=None):
    """
        Plots the silhouette scores for the given embeddings and cluster labels on the provided axis.

        Inputs:
              -ax (matplotlib.axes.Axes): The axis object to plot on.
              -embeddings (np.ndarray): The input data of shape (num_samples, num_features).
              -indx (np.ndarray): The cluster assignments for each sample.
        """
    # Calculate silhouette scores
    silhouette_avg = silhouette_score(embeddings, indx)
    sample_silhouette_values = silhouette_samples(embeddings, indx)

    if ax:
        # Silhouette plot
        n_clusters = np.unique(indx).size
        y_lower = 10
        ax.set_title("Silhouette Plot for Clustering")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster Label")

        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[indx.flatten() == i]
            ith_cluster_silhouette_values.sort()

            y_upper = y_lower + ith_cluster_silhouette_values.shape[0]
            ax.fill_betweenx(np.arange(y_lower, y_upper), ith_cluster_silhouette_values, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * ith_cluster_silhouette_values.shape[0], str(i))

            y_lower = y_upper + 10

        ax.axvline(silhouette_avg, color="red", linestyle="--")
        ax.set_ylim(0, y_lower)
        ax.grid()

    return silhouette_avg, sample_silhouette_values


