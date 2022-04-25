"""Clusters vectorized representations of images using kmeans"""
from sklearn.cluster import KMeans
import numpy as np
import torch
import pandas as pd

def cluster(vectors: np.ndarray, n_clusters = 200, verbose = 1):
    """clusters a list of vectors, O(N)"""
    clusters = KMeans(n_clusters = n_clusters, verbose = verbose).fit_predict(vectors)
    return clusters

def clusters_to_pandas(labels, clusters, savepath = './clusters'):
    df = pd.DataFrame(clusters, index = labels)

    # save dataframe
    df.to_csv(savepath)


if __name__ == '__main__':