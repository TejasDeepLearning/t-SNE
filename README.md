# t-SNE Implementation

This repository contains code that demonstrates the implementation and visualization of t-SNE (t-Distributed Stochastic Neighbor Embedding), a popular dimensionality reduction technique. The code showcases different use cases and applications of t-SNE.

## Table of Contents
- [Overview](#overview)
- [Code Explanation](#code-explanation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
The t-SNE algorithm is a nonlinear dimensionality reduction technique used for visualizing high-dimensional data in lower-dimensional space. It aims to preserve the local structure and relationships between data points, making it useful for exploratory data analysis and pattern discovery. This repository provides examples of how t-SNE can be applied to various datasets and how the results can be visualized.

## Code Explanation
The code in this repository includes several examples of using t-SNE. Here's a brief overview of the different files and their purpose:

1. `xor.py`: Demonstrates t-SNE on synthetic data with multiple clusters. It generates four clusters of random data points and visualizes them using both scatter plots and t-SNE dimensionality reduction.

2. `visualization.py`: Implements t-SNE on a dataset with four distinct blobs. It generates synthetic data and applies t-SNE to visualize the clusters in a lower-dimensional space.

3. `mnist.py`: Uses t-SNE on the Kaggle MNIST dataset. It loads the dataset, reduces the sample size, applies t-SNE, and visualizes the transformed data.

4. `donut.py`: Illustrates t-SNE on a dataset with randomly generated 2-dimensional points. It generates data points from four different distributions and visualizes them using scatter plots and t-SNE.

5. `util.py`: Contains utility functions used in the examples, including functions for activation, error calculation, and weight initialization.

6. `kmeans_clusters.py`: The code demonstrates how K-means clustering can be applied to a 2D dataset and then visualizes the data both in the original feature space and in the t-SNE reduced space. 

## Usage
To run the examples, ensure that you have the required dependencies installed. Each example is contained in a separate file, so you can run them individually by executing the corresponding Python script. For example: <br>
`python example_1.py`

Feel free to modify the code and experiment with different datasets to explore the capabilities of t-SNE.

## Dependencies
The code in this repository relies on the following dependencies:
- `numpy`
- `matplotlib`
- `scipy`
- `scikit-learn`

You can install these dependencies using `pip`: <br>
`pip install numpy matplotlib scipy scikit-learn`
