from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import faiss
import torch


def pairwise_similarity(embedding: np.ndarray, cosine=True, gpu=True):
    """
    computes the pair-wise cosine similarity of a set of data

    args:
        -embedding: the data of shape (N, emb), where N is n° samples and emb is embedding dim to compute the similarity
        -cosine: whether to use cosine similarity, if False uses l2 distance
        -gpu: Whether to use gpu for computation (gpu may be chunked in use)
    return:
        -similarities: The matrix of shape (NxN) with pairwise similarities ordered inside each row in decreasing order
                       i.e. the 1st column for each row it's the max similarity with the closest key
        -idxs: matrix of shape (NxN) where each element is the index of the key which the similarity is computed to
               i.e. for every row you have, at each column, the index of the corresponding key
    """

    max_k = 2048

    embeddings = np.ascontiguousarray(embedding)
    n_features = embeddings.shape[1]
    n_samples = embeddings.shape[0]
    faiss.normalize_L2(embeddings)

    if cosine:
        index = faiss.IndexFlatIP(n_features)
    else:
        index = faiss.IndexFlatL2(n_features)

    if gpu:
        # using gpu
        gpu_resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

        print('looking for similarity...')
        if n_samples > max_k:
            # chunking at maximum K number...

            cols = []

            # creating a partition in chinks (list with dimentions)
            q, r = divmod(n_samples, max_k)
            partition = [max_k for _ in range(q)] + [r] if r != 0 else [max_k for _ in range(q)]
            k_part = 10
            qk, rk = divmod(n_samples, k_part)
            partition_k = [qk for _ in range(k_part)] + [rk] if rk != 0 else [qk for _ in range(k_part)]

            print(f'chunking with {len(partition)} subsets...')

            for i, p in enumerate(partition):
                index.add(embeddings[max_k * i:max_k * i + p, :])
                rows = []
                for j, p1 in enumerate(partition_k):

                    print(f'query {i}/{len(partition)}, key {j}/{len(partition_k)}')
                    s, I = index.search(embeddings[qk * j:qk * j + p1, :], k=p)
                    I = I + max_k*i
                    rows.append((s, I))
                index.reset()
                cols.append((np.concatenate([x for x, _ in rows], 0), np.concatenate([x for _, x in rows], 0)))
            similarities = np.concatenate([x for x, _ in cols], 1)
            idxs = np.concatenate([x for _, x in cols], 1)
        else:
            index.add(embeddings)
            similarities, idxs = index.search(embeddings, k=n_samples)
    else:
        index.add(embeddings)
        similarities, idxs = index.search(embeddings, k=n_samples)

    return similarities, idxs


def cosine_similarity(q, k, corresponding_points=False, gpu=True):
    """

    it computes the cosine similarity between q and k (q * k')

    args:
        -q: query matrix of shape Nxemb where N is n° samples and emb is embedding dim
        -k: key matrix of shape Mxemb where M is n° samples and emb is embedding dim
        -corresponding_points: whether you just want row by row distances (corresponding points)
                               WATCH-OUT: N should be equal to M
        -gpu: whether to use gpu or not (cpu)
    return:
        -dot_products: a list containing the distances between q and k, if corresponding_points=True it is a list of
                       len == N
    """

    accepted_inputs = [np.ndarray, torch.Tensor, list]

    assert q not in accepted_inputs, f'q type {type(q)} not valid, pls use one of: {accepted_inputs}'
    assert k not in accepted_inputs, f'k type {type(k)}not valid, pls use one of: {accepted_inputs}'

    if not torch.is_tensor(q):
        q = torch.tensor(q)
    if not torch.is_tensor(k):
        k = torch.tensor(k)

    if gpu:
        q.to('cuda:0')
        k.to('cuda:0')

    # l2 normalizing each row
    q = torch.nn.functional.normalize(q, p=2, dim=-1)
    k = torch.nn.functional.normalize(k, p=2, dim=-1)

    dot_products = torch.matmul(q, k.T)

    if corresponding_points:
        assert not q.size == k.size, 'q and k has different sizes'
        return list(torch.diagonal(dot_products).numpy())
    else:
        return dot_products


def similarities(q, gpu=True):

    BS = q.size()[0]
    p = q.size()[1]
    c = q.size()[2]

    # q = q.view(q.size(0), -1)  # Shape: [BS, 64*64*256]
    q = torch.nn.functional.normalize(q, p=2, dim=-1)

    # sp = []
    # ch = []
    # im = []

    s = []
    if gpu:
        q.to('cuda:0')

    # print('computing similarity...')
    # # Step 2: Compute the full pairwise similarity matrix (BS x BS)
    # similarity_matrix = torch.matmul(q, q.transpose(0, 1)).cpu().to(torch.float16)  # Shape: [BS, BS]
    # print('similarity computed')

    # for i in range(BS):
    #     for j in range(i + 1, BS):
    #         # mean similarity of images
    #         im.append((similarity_matrix[i, j].item(), i, j))
    #         # mean similarity of pixels of images
    #         sp.append((similarity_matrix[i, j].item() / p, i, j))
    #         # mean similarity of channels of images
    #         ch.append((similarity_matrix[i, j].item() / c, i, j))

    k = q.permute(2, 0, 1).view(c, -1)  # shape [c, p*BS]

    d = []
    max_k = 32

    # chunking similarity
    qu, r = divmod(BS, max_k)
    partition = [max_k for _ in range(qu)] + [r] if r != 0 else [max_k for _ in range(q)]

    print('start chunking...')
    for i in range(BS):
        for n, part in enumerate(partition):
            sim = torch.matmul(q[i], k[:, max_k * n:max_k * n + part*p])  # shape [p, p*part]
            print(f'query {i+1}/{BS}, chunk {n+1}/{len(partition)}')
            sim = sim.permute(1, 0).view(part, p, p)
            diags = torch.diagonal(sim, dim1=1, dim2=2).cpu().tolist()
            if n+1*max_k > i >= n*max_k:  # removing out self similarity
                diags[i - n * max_k] = 10
            d += [(x, i, idx + n * max_k) for idx, x in enumerate(diags) if x != 10]
            # for j in range(part):
            #     if j + n*max_k != i:  # avoiding same batch similarity
            #         for row in range(p):
            #             d.append(([sim[row, row+part*j].item()], i, n*max_k + j))


    # for i in range(BS):
    #     for j in range(i + 1, BS):
    #         print(f'status: {i + 1}/{BS} queries, {j+1}/{BS - i -1} keys...')
    #         try:
    #             d = torch.matmul(q[i], q[j].transpose(0, 1)).to(torch.float16)
    #             similarity_matrix = torch.diagonal(d)
    #             del d
    #         except RuntimeError:
    #             print('too heavy computation for all matmul. Computing on-to-one dot prod...')
    #             similarity_matrix = []
    #             for r in q[i].size[0]:
    #                 similarity_matrix.append(torch.matmul(q[i][r], q[j][r]).to(torch.float16).item())
    #
    #         if torch.is_tensor(similarity_matrix):
    #             s += [(list(similarity_matrix.cpu().numpy()), i, j)]
    #         else:
    #             s += [(similarity_matrix, i, j)]
    #
    #         del similarity_matrix

    return d



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
    sample_silhouette_values = silhouette_score_faiss(embeddings, indx)
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


def compute_elbow(kmeans_model, embeddings, current_K_n, ax, return_D_I=False, spherical=True):
    """
    Computes and plot iteratively the inertia for a given cluster set

    args:
        -kmeans_model: faiss kmeans trained model
        -embeddings: data as (N,emb) to compute inertia respect centroids
        -current_K_n: current n° of centroids
        -ax: matplotlib ax object in which to plot iteratively the result
        -return_D_I: whether to return Distances from clusters and indexes
        -spherical: whether the computed clustering was spherical or euclidean
    return:
        -D: distances of each element from its centroid
        -I: indexes of the corresponding centroid of each element
    """

    print('computing inertia for elbow...')

    if not ax[0].get_title():
        ax[0].set_title('Normalized (mean) Inertia')
        ax[0].set_xlabel("Number of clusters (k)")
        ax[0].set_ylabel("Distortion (Inertia)")
        ax[0].grid()
        ax[1].set_title('Un-normalized Inertia')
        ax[1].set_xlabel("Number of clusters (k)")
        ax[1].set_ylabel("Distortion (Inertia)")
        ax[1].grid()

    embeddings = np.ascontiguousarray(embeddings)
    faiss.normalize_L2(embeddings)

    D, I = kmeans_model.index.search(embeddings, 1)  # Get distances
    if spherical:
        wcss = np.sum(1 - D)/D.shape[0]  # normalized
        wcss_un = np.sum(1 - D)  # unnormalized
    else:
        wcss = np.sum(D**2)/D.shape[0]
        wcss_un = np.sum(1 - D)  # unnormalized
    imbalance = kmeans_model.iteration_stats[-1]['imbalance_factor']
    ax[0].bar(str(f"K = {current_K_n}\n imb:{imbalance: .3f}"), wcss, color='skyblue', edgecolor='black', alpha=0.6)
    ax[1].bar(str(f"K = {current_K_n}\n imb:{imbalance: .3f}"), wcss_un, color='skyblue', edgecolor='black', alpha=0.6)

    if return_D_I:
        return D, I


def silhouette_score_faiss(embeddings, labels, n_neighbors=10, gpu=True):
    """
    Compute Silhouette Score using FAISS for cosine similarity clustering without full similarity matrix.

    Parameters:
    - embeddings: np.ndarray of shape (n_samples, n_features), embeddings for cosine similarity
    - labels: np.ndarray of shape (n_samples,), cluster assignments for each embedding
    - n_neighbors: int, number of nearest neighbors to consider for intra- and inter-cluster distances

    Returns:
    - silhouette_avg: float, average Silhouette Score for all points
    """

    embeddings = np.ascontiguousarray(embeddings)
    faiss.normalize_L2(embeddings)
    n_samples = embeddings.shape[0]
    unique_labels = np.unique(labels)
    # index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner Product index for cosine similarity
    # index.add(embeddings)  # Add embeddings to FAISS index
    #
    if gpu:
        gpu_resources = faiss.StandardGpuResources()
    #     index = faiss.index_cpu_to_gpu(gpu_resources, 0, index)

    silhouette_scores = np.zeros(n_samples)

    samples = half_items(labels)

    for j, (half, label) in enumerate(samples):
        for i in half:
            print(f'{j}/{len(samples)} : {i}/{len(half)}')

            # Find intra-cluster distances (a)
            mask_intra = (labels == label)
            # print(mask_intra, embeddings.shape)
            intra_embeddings = embeddings[np.squeeze(mask_intra), :]
            index_intra = faiss.IndexFlatIP(embeddings.shape[1])
            index_intra.add(intra_embeddings)
            if gpu:
                _, distances_intra = index_intra.search(np.array([embeddings[i]]), min(n_neighbors, intra_embeddings.shape[0]))

            a = np.mean(1 - distances_intra[0, 1:])  # Exclude self distance (0-th position)

            # Find inter-cluster distances (b)
            inter_dists = []
            for other_label in unique_labels:
                if other_label != label:
                    mask_inter = (labels == other_label)
                    inter_embeddings = embeddings[np.squeeze(mask_inter), :]
                    index_inter = faiss.IndexFlatIP(embeddings.shape[1])
                    index_inter.add(inter_embeddings)
                    if gpu:
                        index_inter = faiss.index_cpu_to_gpu(gpu_resources, 0, index_inter)
                    _, distances_inter = index_inter.search(np.array([embeddings[i]]), min(n_neighbors, inter_embeddings.shape[0]))

                    b = np.mean(1 - distances_inter[0])  # Mean distance to this cluster
                    inter_dists.append(b)

            b = min(inter_dists)  # Take the smallest mean distance to another cluster

            # Calculate Silhouette Score for the point
            silhouette_scores[i] = (b - a) / max(a, b)

    # Average silhouette score for all points
    silhouette_avg = np.mean(silhouette_scores)
    return silhouette_avg, silhouette_scores


def half_items(labels):
    unique_labels = np.unique(labels)
    cluster_indices = {label: np.where(labels == label)[0] for label in unique_labels}

    out = []
    # Step 2: Iterate over each cluster, processing a randomly selected half of the samples
    for label, indices in cluster_indices.items():
        # Randomly select half of the samples in the cluster
        n_in_cluster = len(indices)
        half_size = n_in_cluster // 2
        random_indices = np.random.choice(indices, half_size, replace=False)
        out.append((random_indices, label))

    return out


def compute_Kmeans(emb: torch.Tensor, mode: list, clusters: list, min_points_per_centroid=0, niter=20, nredo=1,
                   verbose=True, gpu=True, seed=36, spherical=True, return_Ks=False, save_Ks_path=None,
                   elbow_save_path=None):

    """

    computes Kmeans using faiss

    args:
        - emb: tensor with embeddings as output of sam.get_image_embedding() (Bs, 256, 64, 64)
        - mode: list with mode for clustering:
                 -- pixel   -> clusters considering 256 as features
                 -- channel -> clusters considering 64*64 as features
                 -- image   -> clusters considering 256*64*64 as features
        - clusters: list containing the number of clusters to explore
        - min_points_per_centroid: self explaining I'd say...
        - niter: n° of iterations inside every search
        - nredo: n° of consecutive searches with different random initializations
        - verbose: also self explaining...
        - gpu: whether to use gpu accelleration
        - seed: seed for initialization
        - spherical: whether to use cosine similarity for clustering, if false it uses L2
        - return_Ks: whether to return the centroids' dictionary
        - save_Ks_path: path for saving Ks dictionary (default None => not saving it)
        - elbow_save_path: path for saving elbow plot (default None => not even computing it)

    return:
        - (optional) Ks_full: dictionary containing a dictionary with all centroids x n° clusters divided by modes
    """

    accepted_modes = ['pixel', 'channel', 'image']

    for x in mode:
        assert x not in accepted_modes, f'provided {x} mode is not valid, pls select one/all of {accepted_modes}'

    if gpu and not faiss.get_num_gpus():
        gpu = False

    Ks_full = {}
    for m in mode:

        if return_Ks or save_Ks_path:
            Ks = {}

        if elbow_save_path:
            f, a = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)
            f.suptitle(f'Elbow for {m}-wise cosine distance')
            axs = [a]

        if m == 'pixel':
            embeddings = emb.permute(0, 2, 3, 1).reshape(-1, 256).numpy()
        elif m == 'channel':
            embeddings = emb.reshape(-1, 64 * 64).numpy()
        else:  # image
            embeddings = emb.reshape(emb.size()[0], -1).numpy()

        for i, n in enumerate(clusters):
            if n < embeddings.shape[0]:  # check if the n° of cluster is not bigger that the actual n° of samples
                # Initialize FAISS K-means
                faiss.normalize_L2(embeddings)
                kmeans = faiss.Kmeans(embeddings.shape[-1], n, min_points_per_centroid=min_points_per_centroid,
                                      niter=niter, nredo=nredo, verbose=verbose, gpu=gpu, seed=seed, spherical=spherical)
                # Train the K-means model
                print('running K-means...')
                kmeans.train(embeddings)

                Ks = {str(n): torch.Tensor(kmeans.centroids)}

                if elbow_save_path:
                    # inertia for elbow
                    _, _ = compute_elbow(kmeans, embeddings, n, axs[-1], return_D_I=True)

        if save_Ks_path:
            torch.save(Ks, fr'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\{m}-wise_centroids.pth')

        if elbow_save_path:
            f.savefig(fr'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\{m}-wise_elbow.png', dpi=100,
                      bbox_inches='tight')
        Ks_full[f'{m}-wise'] = Ks
    if return_Ks:
        return Ks_full

