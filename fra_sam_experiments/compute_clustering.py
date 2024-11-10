import numpy as np
import faiss
import torch
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from utils import random_state
from utils.clustering import compute_Kmeans

random_state()

# # Initialize the main application window
# root = tk.Tk()
# root.title("Multiple Matplotlib Figures")

clusters = [2, 4, 10, 100, 200, 500, 1000, 2000]
# clusters = clusters[::-1]

print('loading data...')
emb_dict = torch.load(r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\mmi_2_train_embd.pth')

emb = torch.stack([emb_dict[x] for x in emb_dict.keys()])
print(f" --> {emb.shape}, {emb.dtype}")  # shape [N, 256, 64, 64]


#
# figs = []
# axs = []

# # creatng K figures for siluhettes + 1 for elbow (last)
# for i in range(len(clusters)+1):
#     f, a = plt.subplots(1, 1, figsize=(12, 6))
#     figs.append(f)
#     axs.append(a)
#


# mode = ['pixel', 'channel', 'image']
# for m in mode:
#     Ks = {}
#     f, a = plt.subplots(1, 2, figsize=(19.2, 10.8), dpi=100)
#     f.suptitle(f'Elbow for {m}-wise cosine distance')
#     axs = [a]
#
#     if m == 'pixel':
#         embeddings = emb.permute(0, 2, 3, 1).reshape(-1, 256).numpy()
#     elif m == 'channel':
#         embeddings = emb.reshape(-1, 64*64).numpy()
#     else:  # image
#         embeddings = emb.reshape(emb.size()[0], -1).numpy()
#
#     for i, n in enumerate(clusters):
#         if n < embeddings.shape[0]:  # check if the n° of cluster is not bigger that the actual n° of samples
#             # Initialize FAISS K-means
#             faiss.normalize_L2(embeddings)
#             kmeans = faiss.Kmeans(embeddings.shape[-1], n, max_points_per_centroid=embeddings.shape[0], niter=50,
#                                   nredo=20, verbose=True, gpu=True, seed=36, spherical=True)
#             # Train the K-means model
#             print('running K-means...')
#             kmeans.train(embeddings)
#
#             Ks = {str(n): torch.Tensor(kmeans.centroids)}
#
#             # inertia for elbow
#             _, I = compute_elbow(kmeans, embeddings, n, axs[-1], return_D_I=True)
#
#             # # siluhette
#             # compute_siluhette(embeddings, I, axs[i])
#
#     torch.save(Ks, fr'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\{m}-wise_centroids.pth')
#     f.savefig(fr'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data\{m}-wise_elbow.png', dpi=100, bbox_inches='tight')
#     # plt.show()
#

dst = r'C:\Users\franc\Documents\MedRobotLab\fra_sam_experiments\data'

compute_Kmeans(emb, ['pixel', 'channel', 'image'], clusters, niter=50, nredo=1, save_Ks_path=dst, elbow_save_path=dst)


