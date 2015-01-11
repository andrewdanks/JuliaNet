JuliaNet
========

A deep neural network library implemented in Julia. Features include:

-   Multi-classification
-   Momentum
-   Nesterov's Accelerated Gradient method [3]
-   Multiple hidden layers
-   Mini-batch support
-   Early stopping
-   Custom stop criterion functions
-   Custom activation functions
-   Customized hidden layers
-   Random weight initializing based on fan-in/out heuristics [1]
-   L1/L2 weight penalties
-   Dropout [2]

The library is written such that it is easy to make modifications and
customizations to the network structure and optimization strategies. In
fact, it is encouraged you do make modifications to suit your needs
instead of exclusively treating it as a black box.

Getting started
---------------

See `example.jl` for a typical use case.

Requirements
------------

-   Julia 0.3

There are no other dependencies execept for running the example code,
where you will need
[MNIST.jl](https://github.com/johnmyleswhite/MNIST.jl).

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

