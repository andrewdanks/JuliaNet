JuliaNet
========

A deep neural network library for Julia. Features include:

-   Convolutional and pooling layer support
-   Stacked denoising autoencoders and pre-training
-   Parallelization
-   Dropout [2]
-   Momentum
-   Nesterov's Accelerated Gradient method [3]
-   Multi-classification
-   Online, mini-batch, and batched learning
-   Validation/monitoring sets
-   Serializable models: quit training and reload a model at any time
-   Custom stop criterion functions
-   Custom hyper paramater update criterion
-   Custom activation functions
-   Custom connections between layers
-   Random weight initializing based on fan-in/out heuristics [1]
-   Preprocessing tools: zero mean, unit variance, and more


Getting started
---------------

See the examples directory for use cases.


Requirements
------------

-   Julia 0.3 or above

There are no other dependencies execept for running the example code,
where you will need
[MNIST.jl](https://github.com/johnmyleswhite/MNIST.jl).


Roadmap
-------

Near-future releases want to feature:
-   Better regularization techniques: L2 weight penalties and max incoming weight constraints and scaling
-   Adaptive learning weights
-   More test coverage
-   Command line interface


See the Issues tab for a full list of planned features.


References
----------

[1] Glorot, X. and Bengio, Y. 2010. Understanding the difficulty of
training deep feedforward neural networks. *AISTATS*. (2010).

[2] Hinton, G.E., Srivastava, N., Krizhevsky, A., Sutskever, I. and
Salakhutdinov, R. 2012. Improving neural networks by preventing
co-adaptation of feature detectors. *CoRR*. abs/1207.0580, (2012).

[3] Sutskever, I., Martens, J., Dahl, G.E. and Hinton, G.E. 2013. On the
importance of initialization and momentum in deep learning. *ICML*. 3,
(2013), 1139–1147.

