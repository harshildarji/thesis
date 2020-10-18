# Master's thesis, Uni Passau
Topic: **Investigating Sparsity in Recurrent Neural Networks**

In the past few years, neural networks have evolved from simple **Feedforward Neural Networks** to **Convolutional Neural Networks** (*CNNs*), **Recurrent Neural Networks** (*RNNs*), etc. Where *CNNs* are a perfect fit for tasks where the sequence is not important such as image recognition, *RNNs* are useful when order is important such as machine translation.

An increasing number of layers in a neural network is one of the ways to improve its performance, but it also increases its **complexity** making it much more time and power-consuming  to train. One of the ways to tackle this problem is to introduce **sparsity** in the architecture of the neural network.

**Pruning** is one of the many methods to make a neural network architecture sparse by clipping out weights below a certain threshold while keeping the performance near to the original. Another way is to generate **arbitrary structures** using random graphs and embed them between an input and output layer of an **Artificial Neural Network** (*ANN*).

Many researchers in past years have focused on pruning mainly *CNNs*, while hardly any research is done for the same in the case of *RNNs*. The same also holds in the case of creating sparse architectures for *RNNs* by generating and embedding arbitrary structures. Therefore, this thesis focuses on investigating the effects of the before-mentioned two techniques on the performance of *RNNs*.

We first describe the pruning of *RNNs*, its impact on the performance of *RNNs*, and the number of training epochs required to regain accuracy after the pruning is performed. Next, we continue with the creation and training of **Sparse Recurrent Neural Networks** (*Sparse-RNNs*) and identify the relation between the performance and the graph properties of its underlying arbitrary structure. We perform these experiments on *RNN* with *TANH* nonlinearity (*RNN-TANH*), *RNN* with *ReLU* nonlinearity (*RNN-ReLU*), *GRU*, and *LSTM*. Finally, we analyze and discuss the results achieved from both the experiments.
