"""
Utils for PCA: adapted from the code by Professor Sara Mathieson.
Author: Nasanbayar Ulzii-Orshikh.
Date: 12/30/2020.
"""
import optparse
import sys
import os
import numpy as np
import pickle
import networkx as nx

VECTOR_FILE = "data/strong-house_vectors_i02_10000.txt"
PICKLED_GRAPH = "data/strong-house_pickle"
LABELING_FILE = "data/strong-house_K2_labeling_file.csv"

def parse_args():
    """Parse command line arguments (train and test csv files)."""
    parser = optparse.OptionParser(description='run PCA')

    # Mandatory arguments
    parser.add_option('-m', '--mode', type='string', help='mode: explore or reduce')
    parser.add_option('-n', '--num_of_pcs', type='int', help='number of principal components')

    (opts, args) = parser.parse_args()

    mandatories = ['mode', 'num_of_pcs']
    for m in mandatories:
        if not opts.__dict__[m]:
            print('mandatory option ' + m + ' is missing\n')
            parser.print_help()
            sys.exit()

    return opts

def info_access_vectors():
    matrix = []
    with open(VECTOR_FILE, mode="r") as file:
        for row in file:
            row = row.split(",")
            if row[-1] == "\n":
                row.pop()
            row = [float(i) for i in row]
            matrix.append(row)
    matrix = np.array(matrix)
    return matrix

def dataset_graph():
    with open(PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)
    return graph

def info_access_clustering():
    if not os.path.isfile(LABELING_FILE):
        raise FileNotFoundError("labeling_file not found")

    clustering = {}
    with open(LABELING_FILE, mode="r") as file:
        next(file)
        for row in file:
            row = row.split(",")
            node = int(row[0])
            cluster = int(row[3])
            clustering[node] = cluster
    return clustering

def adjacency_matrix():
    with open(PICKLED_GRAPH, "rb") as file:
        graph = pickle.load(file)
        adj_matrix = nx.convert_matrix.to_numpy_matrix(graph, [i for i in range(len(graph))])
        adj_matrix = np.asarray(adj_matrix)
    return adj_matrix

if __name__ == "__main__":
    adjacency_matrix()
