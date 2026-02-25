import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu


def extract_skeleton(image):
    thresh = threshold_otsu(image)
    binary = image > thresh
    skeleton = skeletonize(binary)
    return skeleton


def skeleton_to_graph(skeleton):
    G = nx.Graph()
    rows, cols = skeleton.shape

    for i in range(rows):
        for j in range(cols):
            if skeleton[i, j]:
                G.add_node((i, j))

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < rows and 0 <= nj < cols:
                            if skeleton[ni, nj]:
                                G.add_edge((i, j), (ni, nj))

    return G


def compute_graph_metrics(G):
    return {
        "Number of Nodes": G.number_of_nodes(),
        "Number of Edges": G.number_of_edges(),
        "Average Degree": sum(dict(G.degree()).values()) / G.number_of_nodes()
        if G.number_of_nodes() > 0 else 0
    }
