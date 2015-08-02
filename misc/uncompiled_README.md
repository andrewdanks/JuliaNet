<!--
To compile:
cat uncompiled_README.md | pandoc --bibliography references.bib --csl citation_style.csl | sed 's/<div class="references">//' | sed 's/<\/div>//' | pandoc -f html -t markdown | sed -e :a -e 's/<[^>]*>//g;/</N;//ba' > README.md
-->
JuliaNet
========

A deep neural network library for Julia. Features include:

-   Multi-classification
-   Convolutional and pooling layer support
-   Autoencoders and pre-training
-   Parallelization support
-   Momentum
-   Nesterov's Accelerated Gradient method [3]
-   Dropout [2]
-   Online, mini-batch, and batched learning
-   Validation/monitoring sets
-   Serializable models: quit training and reload a model at any time
-   Custom stop criterion functions
-   Custom hyper paramater update criterion
-   Custom activation functions
-   Custom connections between layers
-   Random weight initializing based on fan-in/out heuristics [1]


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
