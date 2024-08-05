## Goal
This is my implementation of the original transformer paper ([Vaswani et al.](https://arxiv.org/abs/1706.03762)) in PyTorch.
As of now it'll mostly include the implementation details of the origial paper. However, in the future I might improve it to include explainations for other newcommers

## Techniques Used
While reading/implmenenting the paper I came across a bunch of NLP concepts and algorithms that I wasn't familiar with before.
#### Dropout
Dropout is the zeroing out of random elements of the input vector with probability p during training. This is done to avoid overfitting.
#### Label Smoothing
label smoothing is the same as one-hot encoding but the `1` is substituted by `1 - e` for some 0 < e < 1. And we distribute the `e` among the zeros
#### 
