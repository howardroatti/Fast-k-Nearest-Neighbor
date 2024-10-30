# fkNN - Fast k-Nearest Neighbor Classification

This is a Python implementation of the fkNN (Fast k-Nearest Neighbor) algorithm, based on the paper "Fast k-Nearest Neighbor Classification Using Cluster-Based Trees" by Bin Zhang and S.N. Srihari.

## Introduction

The fkNN algorithm is a variant of the traditional k-Nearest Neighbor (k-NN) classification algorithm. It aims to improve the efficiency and accuracy of k-NN classification, especially on high-dimensional datasets, by leveraging a hierarchical clustering-based tree structure.

The key features of the fkNN algorithm are:

1. Constructing a tree-like data structure, called the "Hyperlevel" and "Plevel", to organize the input data.
2. Using a cluster-based approach to build the tree, which allows for early decision-making and efficient search paths.
3. Incorporating a similarity-based classification mechanism that considers both the similarity between the query point and the decision points in the tree, as well as the labels of the nearest neighbors.

## Algorithm Overview

The fkNN algorithm consists of two main phases:

1. **Tree Construction**:
   - The input data (`X`) is clustered using a chosen clustering method (e.g., K-Means, Agglomerative Clustering).
   - For each cluster, a "Hyperlevel" node is created, containing the indices of the data points in that cluster.
   - The "Hyperlevel" nodes are then recursively grouped into a hierarchical "Plevel" tree structure based on the similarity between the data points.

2. **Classification**:
   - For a given query point, the algorithm traverses the tree, maintaining the `L` best candidate nodes at each level.
   - At the "Hyperlevel", the classification is based on the similarity between the query point and the most dissimilar data point pairs within the node.
   - If there is a tie or the labels of the dissimilar pairs are the same, the classification is made directly at the lower level.
   - Otherwise, a k-NN model is trained on the data points within the "Hyperlevel" node and used to make the final classification.

The algorithm also includes a probability estimation mechanism that considers both the similarity-based weighting and the k-NN probabilities.

## Usage

To use the fkNN implementation, you can follow these steps:

1. Install the required dependencies (e.g., NumPy, scikit-learn).
2. Import the `FkNN` class from the provided code.
3. Instantiate the `FkNN` class with the desired parameters, such as `alpha`, `L`, `k`, and `decisionLevel`.
4. Fit the model using the `fit()` method with your training data (`X`, `y`).
5. Use the `predict()` method to classify new data points.
6. (Optional) Use the `predict_proba()` method to obtain the probability estimates for each class.

Refer to the code comments and docstrings for more detailed usage information.

## Evaluation

The original paper evaluates the fkNN algorithm on the NIST and MNIST datasets, comparing its performance with other k-NN variants. You can use similar datasets and evaluation metrics to assess the performance of your implementation.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to open a new issue or submit a pull request.

## References

1. Bin Zhang and S.N. Srihari, "Fast k-Nearest Neighbor Classification Using Cluster-Based Trees," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 26, no. 4, pp. 525-528, April 2004. doi: 10.1109/TPAMI.2004.1265868
2. [Fast k-Nearest Neighbor Classification Using Cluster-Based Trees](https://www.researchgate.net/publication/8331656_Fast_k-Nearest_Neighbor_Classification_Using_Cluster-Based_Trees)