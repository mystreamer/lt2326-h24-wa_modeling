
import torch
from torch import nn, Tensor
from typing import Tuple
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import os

# local imports
from modules.autoencoder import AutoEncoder
from modules.utils import detect_platform, CLIParser
from modules.wikiart import WikiArtDataset

def extract_embeddings(model, data_loader, device):
    """
    This function gets the "dense" representations from the latent space
    created by the autoencoder.
    """
    model.to(device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0].to(device)
            encoded = model.encoder(inputs)
            # don't forget to detach
            encoded = encoded.cpu().detach()
            embeddings.append(encoded)
    return torch.vstack(embeddings)


def get_label_numbers(data_loader):
    nested_y = []
    for dp in data_loader:
        nested_y.append(dp[1])
    return [int(label) for batch in nested_y for label in batch]


def plot_clusters_2d(data_points, labels, image_path, title="2D Data Clustering", cmap="tab10"):
    """
    Note: This function was generated with some help from generative AI.
    """
    # Convert inputs to np arrays for compatibility
    data_points = np.array(data_points)
    labels = np.array(labels)
    
    # Ensure data points are 2D
    if data_points.shape[1] != 2:
        raise ValueError("data_points must have exactly two dimensions (n_samples, 2).")
    
    # Plot the clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        data_points[:, 0], data_points[:, 1], c=labels, cmap=cmap, s=50, edgecolor='k'
    )
    
    # Add a legend for the clusters
    legend1 = plt.legend(
        *scatter.legend_elements(), title="Clusters", loc="upper right", bbox_to_anchor=(1.15, 1)
    )
    plt.gca().add_artist(legend1)
    
    # Set plot labels and title
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig(image_path)

if __name__ == "__main__":
    # get args
    cp = CLIParser()
    ap = cp.get_argument_parser()
    args = ap.parse_args()

    print(args)

    # Set from config / cli
    save_dir = args.model_save_dir
    batch_size = args.batch_size
    autoencoder_name = args.autoencoder_model_name
    device = detect_platform(args.cuda_num)
    testingdir = args.test_folder
    testdataset = WikiArtDataset(testingdir, device)
    vizdir = args.visualisation_dir
    os.makedirs(vizdir, exist_ok=True)

    # Init model class, load model, put into eval mode
    model = AutoEncoder()
    model.load_state_dict(torch.load(f"./{ save_dir }/{ autoencoder_name }", weights_only=True))
    model.eval()

    # load dataset
    test_loader = torch.utils.data.DataLoader(
        testdataset,
        batch_size=batch_size,
        shuffle=True,
        )

    # Obtain the embeddings for each image 
    embeddings = extract_embeddings(model, test_loader, device)
    
    # Extract labels
    labels = get_label_numbers(test_loader)

    # We know K apriori in the supervised setting
    kmeans = KMeans(n_clusters=len(set(labels)), random_state=42).fit(embeddings.numpy())
    clusters = kmeans.labels_

    # Reduce dimensionality
    pca = PCA(n_components=2)
    embeddings_red = pca.fit_transform(embeddings)

    # Plot and save as image
    plot_clusters_2d(embeddings_red, labels, image_path=f"{ vizdir }/clusters.png")
    
    print(f"Cluster visualisation saved to { vizdir }/clusters.png")