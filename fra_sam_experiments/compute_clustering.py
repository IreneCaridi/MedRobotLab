import numpy as np
import faiss
import torch
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt


from utils import random_state

random_state()



n_clusters = 10

emb_dict = torch.load(r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\mmi_old_train_embd.pth')

embeddings = torch.stack([emb_dict[x].view(emb_dict[x].size()[0], -1) for x in emb_dict.keys()]).view(emb_dict[0].size()[0], -1)

embeddings = embeddings.numpy()

# Initialize FAISS K-means
kmeans = faiss.Kmeans(embeddings.shape[-1], n_clusters, niter=20, nredo=2, verbose=True, gpu=True)

# Train the K-means model
kmeans.train(embeddings)

# Get the centroids of the clusters
centroids = kmeans.centroids

# assignments
D, I = kmeans.index.search(embeddings, 1)


silhouette_avg = silhouette_score(embeddings, I)
sample_silhouette_values = silhouette_samples(embeddings, I)

# Silhouette plot
n_clusters = np.unique(I).size


# DO SOMETHING LIKE THIS FOR ELBOW WHILE COMPUTING CLUSTERS AT DIFFERENT Ks

# wcss = []
# range_n_clusters = range(1, 11)
#
# for n in range_n_clusters:
#     kmeans = faiss.Kmeans(d=embeddings.shape[1], k=n, niter=20, nredo=5)
#     kmeans.train(embeddings)
#
#     # Inertia is the sum of squared distances to the nearest cluster center
#     D, _ = kmeans.index.search(embeddings, 1)  # Get distances
#     wcss.append(np.sum(D ** 2))  # WCSS = sum of squared distances


print("Centroids:\n", centroids)
print("Assigned cluster indices:\n", I.flatten())
