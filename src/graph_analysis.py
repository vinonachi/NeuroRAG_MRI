import numpy as np
import networkx as nx
from skimage.morphology import skeletonize


def extract_skeleton(image):

    binary = image > 0.5
    skeleton = skeletonize(binary)

    return skeleton.astype(float)


def skeleton_to_graph(skeleton):

    G = nx.Graph()

    rows, cols = np.where(skeleton > 0)

    for r, c in zip(rows, cols):
        G.add_node((r, c))

    return G


def compute_graph_metrics(G):

    metrics = {}

    metrics["Number of Nodes"] = G.number_of_nodes()
    metrics["Number of Edges"] = G.number_of_edges()

    if G.number_of_nodes() > 0:
        metrics["Density"] = nx.density(G)
    else:
        metrics["Density"] = 0

    return metrics
