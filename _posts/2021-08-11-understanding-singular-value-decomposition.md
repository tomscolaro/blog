---
title: Understanding Singular Value Decompostion
author: tom
date: 2021-08-11 00:34:00 +0800
categories: [math]
tags: [math, ]
---


# A Powerful Matrix Factorization Technique

In the realm of linear algebra and matrix computations, Singular Value Decomposition (SVD) is a fundamental technique that has widespread applications across various fields. Data analysis, image processing, recommendation systems, and more are all examples. SVD is a powerful tool that allows us to decompose a matrix into its constituent parts, providing valuable insights into the underlying structure and properties of the data.

At its core, SVD is a factorization method that breaks down a given matrix into three distinct matrices, representing its singular values, left singular vectors, and right singular vectors. This decomposition holds significant importance due to its ability to reveal the intrinsic characteristics of the original matrix, facilitating numerous data manipulation and analysis techniques.
<br/>

# As Notation

The formal definition of SVD for an m × n matrix A is as follows:

$$ A = UΣV^T $$

U is an m × m orthogonal matrix whose columns are the left singular vectors.

V is an n × n orthogonal matrix whose columns are the right singular vectors.

Σ is an m × n diagonal matrix containing the singular values of A. The singular values are typically arranged in descending order along the diagonal of Σ.
<br/>

# Meaning of it all

The singular values themselves provide crucial information about the importance or significance of each mode of variation in the data. Larger singular values indicate greater importance, and thus, retaining the top singular values while discarding the rest can lead to effective dimensionality reduction. This property is often exploited in applications such as image compression and data compression, where reducing the dimensionality of the data can lead to more efficient storage and processing.

Additionally, SVD can be leveraged for data approximation or denoising. By setting small singular values to zero, one can eliminate noise or unwanted components from the data, effectively reconstructing a cleaner version of the original matrix. This capability finds applications in signal processing, image denoising, and removing noise from large datasets.

Moreover, SVD plays a central role in recommendation systems, which rely on matrix factorization techniques to analyze user-item interactions. By decomposing a user-item matrix into its constituent parts, SVD allows for the identification of latent factors or features that influence user preferences. These factors can then be utilized to make personalized recommendations or predict missing entries in the matrix.

Another significant application of SVD lies in solving linear systems of equations. Given an overdetermined system (more equations than variables) or an underdetermined system (more variables than equations), SVD can provide a least-squares solution that minimizes the overall error. This property is valuable in various fields, including geodesy, robotics, computer graphics, and many more.

While SVD is a powerful technique, it is worth noting that it can be computationally expensive for large matrices. The computational complexity of the standard SVD algorithm is O(mn^2), which can become impractical for extremely large datasets. However, there exist optimized algorithms and approximate methods that make SVD feasible even for massive matrices.


<br/>

# Calculating SVD as Python Code

## The easy way 

```
import numpy as np

# Define the matrix
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Calculate the SVD
U, S, VT = np.linalg.svd(matrix)

# U: Left singular vectors
print("Left singular vectors:")
print(U)

# S: Singular values
print("Singular values:")
print(S)

# VT: Right singular vectors (transpose)
print("Right singular vectors (transpose):")
print(VT)
```


## The hard way

In this code, the svd() function calculates the SVD of the given matrix using a step-by-step approach. 

Compute the eigenvalues and eigenvectors of A^T A. Let the eigenvalues be λ1, λ2, ..., λr, where r is the rank of A, and let the eigenvectors be v1, v2, ..., vr.

Compute the singular values of A as σi = sqrt(λi), for i = 1, 2, ..., r.

Compute the left singular vectors of A as the normalized eigenvectors of A^T A. Let ui = Avi/σi, for i = 1, 2, ..., r.

Compute the right singular vectors of A as the normalized eigenvectors of AA^T. Let vi = Ai/σi, for i = 1, 2, ..., r.

Complete the SVD of A by assembling the matrices U, Σ, and V^T. The matrix U is formed by stacking the left singular vectors u1, u2, ..., ur as columns. The matrix Σ is a diagonal matrix with the singular values σ1, σ2, ..., σr on the diagonal. The matrix V is formed by stacking the right singular vectors v1, v2, ..., vr as rows, and taking the transpose V^T.
```
import math

def svd(matrix):
    # Step 1: Compute the transpose of the matrix
    transpose = []
    for j in range(len(matrix[0])):
        row = []
        for i in range(len(matrix)):
            row.append(matrix[i][j])
        transpose.append(row)

    # Step 2: Multiply the matrix and its transpose to get A * A^T
    AAT = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix)):
            dot_product = sum(matrix[i][k] * transpose[k][j] for k in range(len(matrix[0])))
            row.append(dot_product)
        AAT.append(row)

    # Step 3: Compute the eigenvalues and eigenvectors of A * A^T
    eigenvalues, eigenvectors = eig(AAT)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    eigenvalues, eigenvectors = zip(*sorted(zip(eigenvalues, eigenvectors), reverse=True))

    # Step 5: Compute the singular values and sort them in descending order
    singular_values = [math.sqrt(eigenvalue) for eigenvalue in eigenvalues]

    # Step 6: Compute the right singular vectors
    V = []
    for eigenvector in eigenvectors:
        norm = math.sqrt(sum(component ** 2 for component in eigenvector))
        V.append([component / norm for component in eigenvector])

    # Step 7: Compute the left singular vectors
    U = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(singular_values)):
            dot_product = sum(matrix[i][k] * V[k][j] for k in range(len(matrix[0])))
            row.append(dot_product / singular_values[j])
        U.append(row)

    # Return the U, S, V matrices
    return U, singular_values, V

# Helper function to compute eigenvalues and eigenvectors of a matrix
def eig(matrix):
    # Placeholder implementation, replace with your own eigendecomposition algorithm
    # Here, we assume the matrix is already diagonal
    eigenvalues = [matrix[i][i] for i in range(len(matrix))]
    eigenvectors = [[1.0 if i == j else 0.0 for j in range(len(matrix))] for i in range(len(matrix))]
    return eigenvalues, eigenvectors

# Define the matrix
matrix = [[1, 2, 3],
          [4, 5, 6],
          [7, 8, 9]]

# Calculate the SVD
U, S, V = svd(matrix)

# U: Left singular vectors
print("Left singular vectors:")
for row in U:
    print(row)

# S: Singular values
print("Singular values:")
print(S)

# V: Right singular vectors (transpose)
print("Right singular vectors (transpose):")
for row in V:
    print(row)
```
<br/>

# Take aways and application 


Singular Value Decomposition (SVD) has a wide range of applications in various fields. Here are some common applications:

* Image Compression: SVD can be used to compress images by representing them in a lower-dimensional space. The singular values determine the importance of each component, and by truncating the less significant singular values, the image can be compressed while preserving important features.

* Recommender Systems: SVD is commonly used in recommender systems to predict user preferences and make personalized recommendations. By decomposing the user-item interaction matrix, SVD can uncover latent factors that capture user preferences and item characteristics.

*    Data Denoising: SVD can be utilized to remove noise from data. By reducing the rank of a noisy data matrix, SVD can filter out the noise and reconstruct a denoised version of the data.

*    Dimensionality Reduction: SVD can be employed for dimensionality reduction in data analysis and machine learning tasks. By selecting the top-k singular values and corresponding singular vectors, SVD can reduce the dimensionality of the data while preserving its essential structure.

*    Natural Language Processing: SVD is used in various natural language processing tasks such as text categorization, document clustering, and latent semantic analysis. It can help identify latent topics and relationships between documents based on their word co-occurrence patterns.

*    Data Visualization: SVD can be used for visualizing high-dimensional data in a lower-dimensional space. By projecting the data onto the singular vectors corresponding to the top singular values, SVD enables visual exploration and analysis of complex datasets.

*    Face Recognition: SVD has been applied to face recognition tasks by representing faces as vectors in a lower-dimensional space. It helps identify the most discriminative features and perform efficient face matching.

*    Collaborative Filtering: SVD is commonly used in collaborative filtering techniques to predict missing ratings in recommendation systems. It can uncover latent factors that capture user preferences and item characteristics, enabling accurate predictions.

<br/>

To conclude, Singular Value Decomposition (SVD) is a versatile matrix factorization technique with diverse applications across multiple domains. By decomposing a matrix into its singular values and corresponding singular vectors, SVD provides a wealth of information about the underlying structure of the data. Whether it's dimensionality reduction, data approximation, recommendation systems, or solving linear systems, SVD proves to be an invaluable tool in the data scientist's arsenal, enabling deeper insights and more efficient data analysis. 
