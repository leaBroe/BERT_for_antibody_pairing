### t-SNE (t-Distributed Stochastic Neighbor Embedding)

t-SNE is a dimensionality reduction technique particularly well-suited for the visualization of high-dimensional datasets. It was developed by [Laurens van der Maaten and Geoffrey Hinton](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf). **T-SNE visualisation of CNN features can help understand how the CNN understands the relationship between different classes which it is trained on.**

### Key Concepts

1. **Dimensionality Reduction:** t-SNE reduces the number of dimensions of the data while preserving its structure as much as possible. It maps high-dimensional data to a lower-dimensional space (usually 2D or 3D) for visualization.

2. **Preserving Local Structure:** t-SNE is designed to keep similar data points close together in the lower-dimensional space, making clusters of similar points more visible and interpretable. **[in the case of a CNN, with the help of t-SNE you can create maps to display which input data seems “similar” for the network.](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a)**

3. **Probabilistic Approach:** t-SNE calculates probabilities that represent pairwise similarities between points in both the high-dimensional and low-dimensional spaces. It tries to minimize the difference (Kullback-Leibler divergence) between these two sets of probabilities.

### How t-SNE Works

1. **Pairwise Similarities in High-Dimensional Space:**
   - For each point, it computes a probability distribution that represents the similarity to other points.
   - This is done using a Gaussian distribution centered at each point. The width of the Gaussian (perplexity) controls the balance between local and global aspects of the data.

2. **Pairwise Similarities in Low-Dimensional Space:**
   - In the low-dimensional space, t-SNE uses a Student’s t-distribution with a single degree of freedom (which has heavier tails than a Gaussian) to compute similarities between points.

3. **Cost Function (KL Divergence):**
   - t-SNE aims to minimize the Kullback-Leibler (KL) divergence between the probability distributions in the high-dimensional and low-dimensional spaces.
   - This is done using gradient descent to iteratively adjust the positions of points in the low-dimensional space.

### Steps Involved in t-SNE

1. **Compute Pairwise Affinities in High-Dimensional Space:**
   - Calculate the similarity between points using a Gaussian distribution.
   - Define the perplexity parameter, which influences the width of the Gaussian.

2. **Compute Pairwise Affinities in Low-Dimensional Space:**
   - Calculate the similarity between points using a Student’s t-distribution.

3. **Optimize Embedding:**
   - Minimize the KL divergence between the high-dimensional and low-dimensional similarities using gradient descent.
   - Adjust points in the low-dimensional space iteratively to improve the embedding.

### Practical Considerations

- **Perplexity:** The perplexity parameter can be seen as a smooth measure of the number of effective nearest neighbors. Typical values range between 5 and 50. The default perplexity value (also used here) from `sklearn.manifold.TSNE` is 30. 
- **Computational Complexity:** t-SNE can be computationally expensive, especially for large datasets. Various optimizations and approximations (e.g., Barnes-Hut t-SNE) exist to speed up the process. Here, the dimensionality was NOT reduced beforehand.
- **Interpretation:** While t-SNE is excellent for visualizing clusters and local structure, the distances in the low-dimensional space should not be over-interpreted. The focus should be on the overall structure and the relative positioning of clusters. The first step in interpreting the plot is to inspect whether there are clusters in the visualization. Clusters indicate the data points that share similar characteristics. Next, we can examine the separation between the clusters. If the clusters overlap, the data points in those clusters share strong similarities. On the other hand, if clusters are clearly distinct, this typically indicates that the dataset contains unambiguously differing groups of data points Similarly, in the t-SNE plot, we can examine the data distribution. Data points clustered closely together typically signify strong correlations. Conversely, when data points scatter randomly across the 2D or 3D space, it suggests a lack of strong relationships within the data. [for a more detailed explanation click here](https://www.baeldung.com/cs/t-distributed-stochastic-neighbor-embedding#:~:text=The%20first%20step%20in%20interpreting,those%20clusters%20share%20strong%20similarities.).

