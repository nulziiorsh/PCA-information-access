"""
Main experimentation pipeline.
Author: Nasanbayar Ulzii-Orshikh.
Date: 12/17/2020.
"""

import os
import pca
import numpy as np
import util
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import csv
import networkx as nx
from scipy import stats
from sklearn.decomposition import PCA
import node2vec_method as n2vr


QUALITATIVE_OUTPUT = "qualitative_analysis/{}_{}_{}_fisher_exact.csv"
PIPELINE_NAMES = []
PIPELINE_IDS = []
BAR_GRAPH_COLOR_PALETTE = ["#BA65A4", "#1A4D68"]

def main():
    opts = util.parse_args()
    mode = opts.mode
    num_of_pcs = opts.num_of_pcs

    mode_valid = 0
    modes = ["info_access", "adjacency", "node2vec", "all"]
    for i in modes:
        if mode == i:
            mode_valid = 1
    if not mode_valid:
        raise ValueError('Not a valid mode; choose from ["info_access", "adjacency", "node2vec", "all"]')

    if num_of_pcs < 1 or num_of_pcs > 3:
        raise ValueError("The number of PCs should be between 1 and 3 inclusively")

    print("\nFetching the representation(s)...")
    representations = []
    if mode == "info_access" or mode == "all":
        matrix = util.info_access_vectors()
        representations.append(("Information Access Signatures", matrix, "info_access"))
        print("Information Access Signatures representation fetched.")
    if mode == "adjacency" or mode == "all":
        matrix = util.adjacency_matrix()
        representations.append(("Adjacency Matrix", matrix, "adjacency"))
        print("Adjacency Matrix representation fetched.")
    if mode == "node2vec" or mode == "all":
        matrix = n2vr.node2vec_matrix()
        representations.append(("Node2Vec", matrix, "node2vec"))
        print("Node2Vec representation fetched.")

    for representation in representations:
        name = representation[0]
        matrix = representation[1]
        id = representation[2]

        print("\n{} representation pipeline started...".format(name))

        print("Input matrix shape: {}".format(matrix.shape))
        PIPELINE_IDS.append(id)
        PIPELINE_NAMES.append(name)
        X = pca_pipeline(matrix, num_of_pcs)

        print("Comparing the similarity between the implemented and scikit-learn PCA methods...")
        pca = PCA(n_components=num_of_pcs)
        sk_matrix = pca.fit_transform(matrix)

        # Equality rate between the two PCA methods in the current pipeline:
        rate = pca_equality(X, sk_matrix)
        print("\tEquality rate for {}: {}".format(name, rate))

        print("{} representation pipeline completed.".format(name))
    return

def pca_pipeline(matrix, num_of_pcs):
    # PCA:
    print("Performing PCA...")
    matrix, eigvalues, eigvectors = pca.compute_pcs(matrix, num_of_pcs)
    print("Projecting onto the PCs...")
    scores_set = pca.scores(matrix, eigvectors)
    print("Plotting by the PCs...")
    pca.plot(scores_set, num_of_pcs, PIPELINE_NAMES[-1], PIPELINE_IDS[-1])
    print("Elbow Method for choosing K clusters...")
    X = elbow_method(scores_set, num_of_pcs, 1, 11)

    num_of_clusters = int(input("Number of clusters? "))
    print("K-Means clustering...")
    labels = kmeans(X, num_of_clusters)
    print("Performing a qualitative analysis...")
    qualitative_analysis(labels, num_of_clusters)
    return X

# Elbow_method code was adapted from https://towardsdatascience.com/k-means-clustering-with-scikit-learn-6b47a369a83c.
def elbow_method(scores_set, num_of_pcs, min_k, max_k):
    """Make elbow graph to choose k hyper-parameter for K-Means clustering."""
    # Build a matrix of (n samples, p features) from the scores_set:
    X = np.concatenate([scores_set[pc] for pc in scores_set], axis=1)
    distortions = []
    for i in range(min_k, max_k):
        kmeans = KMeans(n_clusters=i, random_state=1).fit(X)
        distortions.append(kmeans.inertia_)

    # Plot the elbow graph:
    plt.plot(range(min_k, max_k), distortions, marker='o')
    plt.xticks(range(min_k, max_k))
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.title("Elbow Plot for Clustering via PCA Coordinates\nUsing {}".format(PIPELINE_NAMES[-1]))
    plt.savefig("plots/{}_PCA_{}D_elbow.png".format(PIPELINE_IDS[-1], num_of_pcs), bbox_inches='tight')
    plt.show()
    plt.close()
    return X

def kmeans(X, num_of_clusters):
    # KMeans clustering with random_state for reproducibility.
    labels = KMeans(n_clusters=num_of_clusters, random_state=1).fit_predict(X)
    return labels

def qualitative_analysis(labels, num_of_clusters):
    """Hypothesis testing for whether clustering via PCA reflects the clustering via Information Access."""
    G = util.dataset_graph()

    # Plots bar graphs of "pca_cluster" vs. "info_access_cluster" and "democrat":
    print('\tPlotting composition bar graphs against "info_access_cluster" and "democrat"...')
    pca_clustering = {}
    for i in range(len(labels)):
        pca_clustering[i] = labels[i]
    assign_clusters(G, pca_clustering, "pca_cluster")

    info_access_clustering = util.info_access_clustering()
    assign_clusters(G, info_access_clustering, "info_access_cluster")

    plot_attribute_bar(G, "pca_cluster", "info_access_cluster", num_of_clusters)
    plot_attribute_bar(G, "pca_cluster", "democrat", num_of_clusters)

    # Performs Fisher Exact tests against "info_access_cluster" and "democrat" if K == 2:
    # that is, if the number of clusters from the user input is equal to the "ground truth"
    # number of clusters, found via Information Access Signatures.
    if num_of_clusters == 2:
        print('\tRunning Fisher Exact tests against "info_access_cluster" and "democrat"...')
        fisher_exact(G, "pca_cluster", "info_access_cluster")
        fisher_exact(G, "pca_cluster", "democrat")
    return

def assign_clusters(G, clustering, attribute):
    nx.set_node_attributes(G, clustering, attribute)
    return

def plot_attribute_bar(graph, cluster_method, attribute, num_of_clusters):
    """Plots a bar graph for the composition of the attribute in each cluster"""
    # Holder for categorical values of the attribute: when we take a set of it,
    # we can determine the nodes' values without hard-coding them.
    nodes = []
    clusters_total = {cluster: [] for cluster in range(num_of_clusters)}
    no_attribute_dict = {cluster: 0.0 for cluster in range(num_of_clusters)}
    for node in graph.nodes:
        node_cluster = graph.nodes[node][cluster_method]
        try:
            if graph.nodes[node][attribute] is not None:
                # Since the data type is categorical, there is no need to convert to int or float or take log.
                value = graph.nodes[node][attribute]

                # Relabels the binary attribute to the attribute name itself for interpretability.
                if "cluster" not in attribute:
                    if value == 0 or value == "0" or value == "False" or value is False:
                        value = "not {}".format(attribute)
                    elif value == 1 or value == "1" or value == "True" or value is True:
                        value = attribute

                clusters_total[node_cluster].append(value)
                nodes.append(value)
        except:
            no_attribute_dict[node_cluster] += 1

    # Error conditions:
    no_attr_count = 0
    for attr in no_attribute_dict:
        no_attr_count += no_attribute_dict[attr]
    if no_attr_count != 0:
        raise ValueError("Non-zero no_attribute_dict")
    total_size = len(nodes)
    if total_size == 0:
        raise ValueError("Zero nodes")

    set_of_attr_values = set(nodes)
    list_of_attr_values = sorted(set_of_attr_values)
    x_values = [i for i in range(num_of_clusters)]

    attr_sections = {}
    for attr_value in list_of_attr_values:
        y_values = [clusters_total[a_cluster].count(attr_value)/len(clusters_total[a_cluster]) for a_cluster in x_values]
        attr_sections[attr_value] = y_values

    offset = [0 for i in x_values]
    for_legend_values = []
    for_legend_labels = []
    color_counter = 0
    for attr_value in list_of_attr_values:
        bar_container_object = plt.bar(x_values, attr_sections[attr_value], bottom=offset, color=BAR_GRAPH_COLOR_PALETTE[color_counter])
        for_legend_values.append(bar_container_object[0])
        for_legend_labels.append(attr_value)
        offset = np.add(offset, attr_sections[attr_value]).tolist()
        color_counter += 1

    plt.xlabel('Clusters')
    plt.xticks(x_values)
    plt.ylabel('Probability')
    plt.legend(for_legend_values, for_legend_labels)

    plt.title('Frequency of "{}" Across "{}" Clusters\nUsing {}'.format(attribute, cluster_method, PIPELINE_NAMES[-1]))
    plt.savefig("plots/{}_BG_K{}_{}_vs_{}.png".format(PIPELINE_IDS[-1], num_of_clusters, cluster_method, attribute),
                bbox_inches='tight')
    plt.close()
    return

# Fisher Exact Test was adapted from my own code from Beilinson et al. (2020).
def fisher_exact(G, ground_attribute, interest_attribute):
    with open(QUALITATIVE_OUTPUT.format(PIPELINE_IDS[-1], ground_attribute, interest_attribute), 'w') as file:
        fieldnames = ["p_value", "contingency_table"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        user_obj_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
        fisher_exact_helper(G, user_obj_writer, ground_attribute, interest_attribute)
    return

def fisher_exact_helper(graph, user_obj_writer, ground_attribute, interest_attribute):
    contingency_table = {interest_attribute: [0, 0], "not-" + interest_attribute: [0, 0]}
    for node_int in range(len(graph.nodes)):
        attribute = graph.nodes[node_int][interest_attribute]
        if attribute == "True" or attribute is True or attribute == 1 or attribute == "1":
            attribute = interest_attribute
        elif attribute == "False" or attribute is False or attribute == 0 or attribute == "0":
            attribute = "not-" + interest_attribute
        cluster = graph.nodes[node_int][ground_attribute]
        contingency_table[attribute][cluster] += 1

    odds_ratio, p_value = stats.fisher_exact([contingency_table[i] for i in contingency_table])

    row = [p_value, contingency_table]
    user_obj_writer.writerow(row)
    return

def pca_equality(X, sk_matrix):
    incorrect = 0
    total = 0
    for i in range(len(X)):
        for j in range(len(X[i])):
            if abs(sk_matrix[i][j]) != X[i][j]:
                incorrect += 1
            total += 1
    return incorrect / total

if __name__ == "__main__":
    main()
