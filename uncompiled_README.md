<!--
To compile:
cat uncompiled_README.md | pandoc --bibliography references.bib --csl citation_style.csl | sed 's/<div class="references">//' | sed 's/<\/div>//' | pandoc -f html -t markdown | sed -e :a -e 's/<[^>]*>//g;/</N;//ba' > README.md
-->
JuliaNet
========

A deep neural network library implemented in Julia. Some features include:

-   Multi-classification
-   Momentum
-   Nesterov's Accelerated Gradient method [@sutskever:2013:nag]
-   Mini-batch support
-   Convolutional layers
-   Subsampling (pooling) layers
-   Input data with multiple channels (e.g., RGB images)
-   Custom stop criterion functions (i.e., early stopping)
-   Custom activation functions
-   Customized hidden layers
-   Random weight initializing based on fan-in/out heuristics [@xavier:2010:weightinit]
-   Dropout [@hinton:2012:dropout]
-   Hyper parameter update hooks

The library is written such that it is easy to make modifications and customizations to the network structure and optimization strategies. In fact, it is encouraged you do make modifications to suit your needs instead of exclusively treating it as a black box.


Getting started
---------------

See the `examples` directory for a variety of use cases.


Requirements
------------

-   Julia 0.3 or above
-   Devectorize
-   PyCall
-   SciPy/NumPy

SciPy/NumPy and PyCall hopefully won't be requirements in the future, but they are currently used for efficiently computing N-dimensional convolutions which Julia currently doesn't support.

For running the example code, you may need the following:
-   [Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
-   [MNIST.jl](https://github.com/johnmyleswhite/MNIST.jl) d

Roadmap
-------

Near-future releases plan to support:

-   L1/L2 weight penalties
-   Max incoming weight constraints and scaling

See the Issues tab for a full list of planned features.


References
----------
