from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import faiss


def silhouette(data, labels):
    """
    Compute silhouette scores for the given data points based on cosine similarity.

    Parameters:
    - data (numpy.ndarray): The input data, shape (num_samples, num_features).
    - labels (numpy.ndarray): The cluster labels for each data point, shape (num_samples,).

    Returns:
    - silhouette_scores (numpy.ndarray): The silhouette scores for each data point.
    """

    # Normalize the data to unit vectors for cosine similarity

    labels = labels.flatten()
    data = np.ascontiguousarray(data)
    faiss.normalize_L2(data)

    # Create FAISS index for inner product (cosine similarity)
    index = faiss.IndexFlatIP(data.shape[1])
    index.add(data)

    n_samples = len(data)

    # Calculate pairwise similarities
    if not n_samples > 100000:
        similarities, _ = index.search(data, k=n_samples)
    else:
        bs = 100000
        n_samples = len(data)
        print(f'there not enough GPU mem, using batches of {bs} samples as queries')
        similarities = np.zeros((n_samples, n_samples), np.float32)
        for i in range(0, n_samples, bs):

            current_batch = data[i:min(i+bs, n_samples), :]

            print(current_batch.shape)

            # Compute distances between the current batch and all data
            distances, _ = index.search(current_batch, k=n_samples)  # Search for all points

            # Store the computed distances in the distance_matrix
            similarities[i:min(i+bs, n_samples), :] = distances

    # Compute silhouette scores
    silhouette_scores = np.zeros(len(data))
    unique_labels = np.unique(labels)

    for i in range(len(data)):
        # Get the label of the current point
        label_i = labels[i]

        # Distances to all other points
        sim_i = similarities[i]

        # Get same-cluster and different-cluster indices
        same_cluster_indices = np.where(labels == label_i)[0]
        different_cluster_indices = np.where(labels != label_i)[0]

        # Calculate a(i): mean distance to points in the same cluster
        if len(same_cluster_indices) > 1:
            a_i = np.mean(sim_i[same_cluster_indices[same_cluster_indices != i]])
        else:
            a_i = 0.0  # If there's only one point in the cluster

        # Calculate b(i): mean distance to points in the nearest different cluster
        if len(different_cluster_indices) > 0:
            # Initialize b_i with a large value to find the minimum
            b_i = float('inf')

            for label_b in unique_labels:
                if label_b != label_i:
                    # Find points belonging to the different cluster
                    points_in_different_cluster = different_cluster_indices[labels[different_cluster_indices] == label_b]
                    if len(points_in_different_cluster) > 0:
                        # Calculate mean similarity to points in this different cluster
                        mean_similarity = np.mean(sim_i[points_in_different_cluster])
                        # Update b_i to be the minimum mean similarity found
                        b_i = min(b_i, mean_similarity)
            # If no valid b_i found, set it to 0 (or any appropriate value)
            if b_i == float('inf'):
                b_i = 0.0
        else:
            b_i = 0.0  # No different clusters

        # Calculate silhouette score for point i
        silhouette_scores[i] = (b_i - a_i) / max(a_i, b_i) if max(a_i, b_i) > 0 else 0.0

    return silhouette_scores


def compute_siluhette(embeddings, indx, ax=None, return_values=False):
    """
        Plots the silhouette scores for the given embeddings and cluster labels on the provided axis.

        Inputs:
              -ax (matplotlib.axes.Axes): The axis object to plot on.
              -embeddings (np.ndarray): The input data of shape (num_samples, num_features).
              -indx (np.ndarray): The cluster assignments for each sample.
        """

    print('computing siluhette...')

    # Calculate silhouette scores
    sample_silhouette_values = silhouette(embeddings, indx)
    silhouette_avg = np.mean(sample_silhouette_values)

    if ax:
        # Silhouette plot
        n_clusters = np.unique(indx).size
        y_lower = 10
        if not ax.get_title():
            ax.set_title("Silhouette Plot for Clustering")
            ax.set_xlabel("Silhouette Coefficient")
            ax.set_ylabel("Cluster Label")

        cluster_silhouette_values = []
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[indx.flatten() == i]
            ith_cluster_silhouette_values.sort()
            cluster_silhouette_values.append(np.mean(ith_cluster_silhouette_values))

        #     y_upper = y_lower + ith_cluster_silhouette_values.shape[0]
        #     ax.fill_betweenx(np.arange(y_lower, y_upper), ith_cluster_silhouette_values, alpha=0.7)
        #     ax.text(-0.05, y_lower + 0.5 * ith_cluster_silhouette_values.shape[0], str(i))
        #
        #     y_lower = y_upper + 10
        #
        # print(silhouette_avg.shape)
        # ax.axvline(silhouette_avg, color="red", linestyle="--")
        # ax.set_ylim(0, y_lower)
        # ax.grid()
        # Prepare to plot horizontal bars
        bars = ax.barh(range(n_clusters), cluster_silhouette_values, alpha=0.7, color='skyblue', edgecolor='black')
        for bar, label in zip(bars, range(n_clusters)):

            ax.text(0.001, bar.get_y() + bar.get_height() / 2, label, va='center', ha='right', color='black',
                    fontsize=10)
        ax.axvline(silhouette_avg, color='red', linestyle='--', label='Average silhouette score')
        ax.axvline(0, color='black')
        ax.legend()

    if return_values:
        return silhouette_avg, sample_silhouette_values


def compute_elbow(kmeans_model, embeddings, wcss, current_K_n, ax, return_D_I=False, spherical=True):

    print('computing inertia for elbow...')

    if not ax.get_title():
        ax.set_title("Elbow Method for Optimal k")
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel("Distortion (Inertia)")
        ax.grid()

    D, I = kmeans_model.index.search(embeddings, 1)  # Get distances
    if spherical:
        wcss.append(np.sum(D))  # /D.shape[0]
    else:
        wcss.append(np.sum(D**2))
    ax.bar(str(f"K = {current_K_n}"), wcss[-1], color='skyblue', edgecolor='black', alpha=0.6)

    if return_D_I:
        return D, I



