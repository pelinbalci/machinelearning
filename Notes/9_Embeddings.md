### Embeddings

* Ref_1: https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture
* Ref_2: https://www.youtube.com/watch?v=JGHVJXP9NHw (Embeddings for Everything: Search in the Neural Network Era)


Embeddings: translate large sparse vectors into a lower-dimensional space that preserves semantic relationships.

Embeddings can be learned from data.

No separate training process needed -- the embedding layer is just a hidden layer with one unit per dimension.


### Embedding lookup as matrix multiplication
The lookup, multiplication and addition procedure we've just described is equivalent to matrix multiplication. 

    Given a 1 X N sparse representation S and an N X M embedding table E, 
    
    the matrix multiplication S X E gives you the 1 X M dense vector.
    
### But how do you get E in the first place?

- Standard Dimensionality Reduction Techniques

Example: PCA

- Word2vec

More info: https://www.tensorflow.org/tutorials/text/word_embeddings


### Selecting Dimensions of Embeddings

* Higher-dimensional embeddings can more accurately represent the relationships between input values.
* But more dimensions increases the chance of OVERFITTING and leads to SLOWER training.
* Empirical rule-of-thumb (a good starting point but should be tuned using the validation data):

    dimensions = (possible values)^(1/4)


### Embeddings in Neural Network

![image_description](/Users/pelin.balci/PycharmProjects/machinelearning/Notes/embedding_1.png)

![image_description](/Users/pelin.balci/PycharmProjects/machinelearning/Notes/embedding_2.png)

![image_description](/Users/pelin.balci/PycharmProjects/machinelearning/Notes/embedding_3.png)
