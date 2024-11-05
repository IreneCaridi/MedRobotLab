import numpy as np
import faiss
import torch
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import random_state
from utils.clustering import compute_siluhette, compute_elbow

random_state()

# # Initialize the main application window
# root = tk.Tk()
# root.title("Multiple Matplotlib Figures")

clusters = [3, 10, 100, 200, 500, 1000]
clusters = clusters[::-1]

print('loading data...')
emb_dict = torch.load(r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\mmi_old_train_embd.pth')

embeddings = torch.stack([emb_dict[x].view(emb_dict[x].size()[0], -1)
                          for x in emb_dict.keys()]).view(emb_dict[list(emb_dict.keys())[0]].size()[0], -1)

# permuting to get Nx256 (256 = embedding size)
embeddings = embeddings.permute(1, 0).numpy()
print(f" --> {embeddings.shape}")


figs = []
axs = []

# creatng K figures for siluhettes + 1 for elbow (last)
for i in range(len(clusters)+1):
    f, a = plt.subplots(1, 1, figsize=(12, 6))
    figs.append(f)
    axs.append(a)


wcss = []
for i, n in enumerate(clusters):
    # Initialize FAISS K-means
    kmeans = faiss.Kmeans(embeddings.shape[-1], n, max_points_per_centroid=embeddings.shape[0], niter=100,
                          nredo=1, verbose=True, gpu=True, seed=36, spherical=True)
    # Train the K-means model
    print('running K-means...')
    kmeans.train(embeddings)

    # inertia for elbow
    _, I = compute_elbow(kmeans, embeddings, wcss, n, axs[-1], return_D_I=True)

    # siluhette
    compute_siluhette(embeddings, I, axs[i])

plt.show()

# # Create a canvas to display the first figure
# canvas1 = FigureCanvasTkAgg(fig1, master=root)
# canvas1_widget = canvas1.get_tk_widget()
# canvas1_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# # Get the centroids of the clusters
# centroids = kmeans.centroids
#
# # assignments
# D, I = kmeans.index.search(embeddings, 1)


# DO SOMETHING LIKE THIS FOR ELBOW WHILE COMPUTING CLUSTERS AT DIFFERENT Ks

# wcss = []
# clusters = range(1, 11)
#
# for n in clusters:
#     kmeans = faiss.Kmeans(d=embeddings.shape[1], k=n, niter=20, nredo=5)
#     kmeans.train(embeddings)
#
#     # Inertia is the sum of squared distances to the nearest cluster center
#     D, _ = kmeans.index.search(embeddings, 1)  # Get distances
#     wcss.append(np.sum(D ** 2))  # WCSS = sum of squared distances

#
# print("Centroids:\n", centroids.shape)
# print("Assigned cluster indices:\n", I.flatten().shape)

