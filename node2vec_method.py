"""
Implementation of the node2vec method, used to create a new representation of the dataset.
Author: Nasanbayar Ulzii-Orshikh.
Date: 12/28/2020.
"""

from node2vec import Node2Vec
import util
import numpy as np

def node2vec_matrix():
    print("Transforming the dataset by node2vec...")
    graph = util.dataset_graph()
    node2vec = Node2Vec(graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    matrix = []
    for i in range(len(graph)):
        matrix.append(model.wv.get_vector(str(i)))

    model.wv.save_word2vec_format("word2vec_embedding")
    model.save("word2vec_model")
    return np.array(matrix)

if __name__ == "__main__":
    node2vec_matrix()
