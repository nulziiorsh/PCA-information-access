"""
Implementation and correctness analysis of Principal Component Analysis (PCA).
Author: Nasanbayar Ulzii-Orshikh.
Date: 12/17/2020.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import util
from mpl_toolkits import mplot3d

def compute_pcs(matrix, num_of_pcs):
    # Centering the dataset:
    print("\tCentering the dataset...")
    col_means = np.mean(matrix, axis=0)
    matrix = matrix - col_means

    # Covariance matrix:
    print("\tComputing the covariance matrix...")
    cov_matrix = np.cov(matrix, rowvar=False)

    # Eigendecomposition of the covariance matrix:
    print("\tPerforming eigendecomposition for {} PCs...".format(num_of_pcs))
    eigvalues, eigvectors = np.linalg.eig(cov_matrix)
    eigvectors = np.transpose(np.array(eigvectors))
    order = {eigvalues[i]: i for i in range(len(eigvalues))}
    eigvalues_sorted = sorted(eigvalues, reverse=True)
    eigvectors_sorted = [eigvectors[order[value]].reshape((len(eigvectors[order[value]]), 1)) for value in eigvalues_sorted]
    print("\tPCs computed.")
    return matrix, eigvalues_sorted[:num_of_pcs], eigvectors_sorted[:num_of_pcs]

def scores(matrix, eigvectors):
    scores_set = {}
    for i in range(len(eigvectors)):
        scores_set["PC{}".format(i + 1)] = np.matmul(matrix, eigvectors[i])
    return scores_set

def plot(scores_set, num_of_pcs, name, id):
    # Colors categorized by index: 0 - republican/red, 1 - democrat/blue:
    colors = np.array(["red", "blue"])
    parties = np.array(["Republican", "Democrat"])

    coordinates = []
    for i in range(len(scores_set)):
        coordinates.append(scores_set["PC{}".format(i + 1)])
    if len(coordinates) == 1:
        coordinates.append([0 for j in scores_set["PC1"]])
    split_coordinates = split_by_party(coordinates)

    if len(scores_set) == 1 or len(scores_set) == 2:
        fig, ax = plt.subplots()
        for i in range(len(split_coordinates)):
            ax.scatter(split_coordinates[i][0], split_coordinates[i][1], c=colors[i], label=parties[i])

    if len(scores_set) == 1:
        plt.xlabel("PC1")
    elif len(scores_set) == 2:
        plt.xlabel("PC1")
        plt.ylabel("PC2")
    elif len(scores_set) == 3:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        for i in range(len(split_coordinates)):
            ax.scatter3D(split_coordinates[i][0], split_coordinates[i][1], split_coordinates[i][2], c=colors[i], label=parties[i])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
    plt.legend()
    plt.title("PCA of 114th Congressional Co-sponsorship Network Dataset\nUsing {}".format(name))
    plt.savefig("plots/{}_PCA_{}D_cosponsorship.png".format(id, num_of_pcs))
    plt.show()
    plt.close()
    return

def split_by_party(a_list):
    # a_list is in form [x, y, z]
    # coordinates[0] are republican/red; coordinates[1] are democrat/blue
    coordinates = [[], []]
    for i in range(len(a_list)):
        for j in range(len(coordinates)):
            coordinates[j].append([])

    G = util.dataset_graph()
    # For each node:
    for i in range(len(G.nodes)):
        node_index = i
        # Decide if republican or democrat by 0 or 1 at "democrat" attribute:
        party = int(G.nodes[node_index]["democrat"])
        # For each dimension of coordinates passed in the argument:
        for j in range(len(a_list)):
            dimension = j
            coordinates[party][dimension].append(a_list[dimension][node_index])
    return coordinates
