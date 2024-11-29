import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

# Function to calculate statistics
def calculate_statistics(data):
    return {
        "mean": data.mean(),
        "median": data.median(),
        "q1": data.quantile(0.25),
        "q2": data.median(),  # Same as median
        "q3": data.quantile(0.75)
    }

# Carica i due file CSV
file1 = "../image/dataset_mmi/IoU/iou_mask_bbox.csv"
file2 = "../image/dataset_mmi/IoU/iou_mask_pts.csv"
file3 = "../image/dataset_mmi/IoU/iou_mask_three_pts_2.csv"

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df3 = pd.read_csv(file3)

# Unire i tre DataFrame per confronto
merged_df = pd.merge(df1, df2, on=["image", "class"], suffixes=("_method1", "_method2"))
merged_df = pd.merge(merged_df, df3, on=["image", "class"])
merged_df.rename(columns={"iou": "iou_method3"}, inplace=True)

# Calcolare differenze tra IoU
merged_df["iou_diff_1_2"] = merged_df["iou_method1"] - merged_df["iou_method2"]
merged_df["iou_diff_1_3"] = merged_df["iou_method1"] - merged_df["iou_method3"]
merged_df["iou_diff_2_3"] = merged_df["iou_method2"] - merged_df["iou_method3"]

# Statistiche sulle IoU
stats = {
    "mean": merged_df[["iou_method1", "iou_method2", "iou_method3"]].mean(),
    "median": merged_df[["iou_method1", "iou_method2", "iou_method3"]].median(),
    "std_dev": merged_df[["iou_method1", "iou_method2", "iou_method3"]].std()
}

# Percentuale di maschere sopra una soglia IoU
threshold = 0.5
above_threshold = {
    "bbox": (merged_df["iou_method1"] > threshold).mean() * 100,
    "center pts": (merged_df["iou_method2"] > threshold).mean() * 100,
    "three pts": (merged_df["iou_method3"] > threshold).mean() * 100,
}

# Confronto: quale metodo è il migliore più spesso
merged_df["best_method"] = merged_df[["iou_method1", "iou_method2", "iou_method3"]].idxmax(axis=1)
best_counts = merged_df["best_method"].value_counts()

# Visualizzare distribuzioni delle IoU
plt.figure(figsize=(12, 8))
plt.hist(merged_df["iou_method1"], bins=20, alpha=0.5, label="bbox")
plt.hist(merged_df["iou_method2"], bins=20, alpha=0.5, label="center pts")
plt.hist(merged_df["iou_method3"], bins=20, alpha=0.5, label="three pts")
plt.axvline(x=threshold, color='r', linestyle='--', label="Threshold")
plt.title("IoU Distribution for Three Methods")
plt.xlabel("IoU")
plt.ylabel("Frequency")
plt.legend()
plt.savefig("iou_distribution_comparison_2.png")
plt.show()

# Risultati a schermo
print(f"Statistiche delle IoU:\n{stats}")
print(f"Percentuale sopra soglia {threshold}:\n{above_threshold}")
print(f"Metodo migliore in più casi:\n{best_counts}")

# Unique classes
classes = merged_df["class"].unique()

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Loop over each class and plot
for idx, cls in enumerate(classes):
    class_data = merged_df[merged_df["class"] == cls]
    
    # Plot histograms for IoU of each method
    axes[idx].hist(class_data["iou_method1"], bins=20, alpha=0.5, label="bbox")
    axes[idx].hist(class_data["iou_method2"], bins=20, alpha=0.5, label="center pts")
    axes[idx].hist(class_data["iou_method3"], bins=20, alpha=0.5, label="three pts")
    
    # Add titles and labels
    axes[idx].set_title(f"Class {cls} IoU Distribution")
    axes[idx].set_ylabel("Frequency")
    axes[idx].legend()

# Set common X-axis label
plt.xlabel("IoU")
plt.tight_layout()
plt.savefig("classwise_iou_comparison_2.png") 
plt.show()

# Create a general box plot
data = [
    merged_df["iou_method1"],
    merged_df["iou_method2"],
    merged_df["iou_method3"]
]

plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=["bbox", "center pts", "three pts"])
plt.title("IoU Distribution for All Methods")
plt.ylabel("IoU")
plt.xlabel("Methods")
plt.savefig("iou_boxplot_comparison_2.png")
plt.show()

all_stats = {}
# Create subplots for class-wise box plots
fig, axes = plt.subplots(len(classes), 1, figsize=(10, 15), sharex=True)

# Loop through classes to create box plots
for idx, cls in enumerate(classes):
    class_data = merged_df[merged_df["class"] == cls]
    data = [
        class_data["iou_method1"],
        class_data["iou_method2"],
        class_data["iou_method3"]
    ]
    
    # Calculate statistics for each method
    class_stats = {}
    for method_idx, method_data in enumerate(data):
        class_stats[f"Method {method_idx + 1}"] = calculate_statistics(method_data)
    
    all_stats[f"Class {cls}"] = class_stats

    axes[idx].boxplot(data, labels=["bbox", "center pts", "three pts"])
    axes[idx].set_title(f"Class {cls} IoU Distribution")
    axes[idx].set_ylabel("IoU")

# Set common X-axis label
plt.xlabel("Methods")
plt.tight_layout()
plt.savefig("classwise_iou_boxplot_comparison_2.png")
plt.show()

# Display statistics
for cls, stats in all_stats.items():
    print(f"Statistics for {cls}:")
    for method, values in stats.items():
        print(f"  {method}:")
        print(f"    Mean IoU: {values['mean']:.4f}")
        print(f"    Median IoU: {values['median']:.4f}")
        print(f"    Q1 (25%): {values['q1']:.4f}")
        print(f"    Q2 (50%): {values['q2']:.4f}")
        print(f"    Q3 (75%): {values['q3']:.4f}")
    print()