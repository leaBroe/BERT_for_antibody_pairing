### UMAP (Uniform Manifold Approximation and Projection)

UMAP is a non-linear dimensionality reduction technique developed by [Leland McInnes, John Healy, and James Melville](https://arxiv.org/pdf/1802.03426), UMAP is known for its ability to preserve both local and global structure in the data. It is often used as an alternative to t-SNE due to its efficiency and scalability.

1. **Manifold Learning:** UMAP assumes that the data lies on a manifold (a continuous, smooth surface) within a higher-dimensional space and aims to learn the manifold structure.
2. **Topological Data Analysis:** UMAP uses techniques from topological data analysis to build a graphical representation of the data.
3. **Preserving Local and Global Structure:** UMAP aims to maintain both local neighborhood relationships and the global data structure when projecting to a lower-dimensional space.

### How UMAP Works

1. **Constructing the Fuzzy Topological Representation:**
    - **Nearest Neighbors:** For each data point, UMAP identifies a fixed number of nearest neighbors using a metric like Euclidean distance.
    - **Fuzzy Simplicial Complex:** UMAP constructs a weighted graph (simplicial complex) where each point is connected to its neighbors with edge weights representing the strength of their relationship.

2. **Constructing the Low-Dimensional Representation:**
    - **Optimization:** UMAP uses stochastic gradient descent to optimize the layout of the points in the lower-dimensional space. It aims to minimize the difference between the high-dimensional fuzzy simplicial complex and its low-dimensional counterpart.
    - **Cost Function:** The cost function encourages points that are close in high-dimensional space to be close in the low-dimensional space, while allowing distant points to be further apart.

### UMAP Parameters

- **`n_neighbors`:** Determines the local neighborhood size used for manifold approximation. Larger values capture more global structure, while smaller values focus on local detail.
- **`min_dist`:** Controls the minimum distance between points in the low-dimensional space. Smaller values preserve more local structure but can lead to densely packed clusters.
- **`n_components`:** The number of dimensions to reduce the data to (typically 2 or 3 for visualization).

### UMAP for CNN Embeddings

CNN embeddings are high-dimensional vectors representing learned features from a neural network. Applying UMAP to these embeddings can help in visualizing and understanding the complex relationships within the data.

### Assumptions and Interpretation

1. **Local and Global Structure:** UMAP assumes that the data has both meaningful local neighborhoods and a global manifold structure that can be captured in the low-dimensional space.
2. **Manifold Learning:** UMAP is based on the assumption that high-dimensional data lies on a lower-dimensional manifold. It seeks to unfold this manifold into a lower-dimensional space.
3. **Interpretation of Overlap:** Overlap in UMAP plots may indicate inherent overlaps in the high-dimensional space or insufficient separation between clusters. It can also reflect complex relationships that are not purely linear.

### Comparison with PCA

- **Linear vs. Non-linear:** Unlike PCA, which only captures linear relationships, UMAP captures non-linear relationships and is more effective for complex, high-dimensional data.
- **Preservation of Structure:** UMAP aims to preserve both local and global structure, while PCA primarily focuses on maximizing variance capture, which may not always correspond to meaningful clusters.

[For more information about UMAP click here](https://alleninstitute.org/resource/what-is-a-umap/#:~:text=UMAPs%20are%20helpful%20ways%20of,an%20x%20and%20y%20graph.)