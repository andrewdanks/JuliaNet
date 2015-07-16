<!--
To compile:
cat uncompiled_README.md | pandoc --bibliography references.bib --csl citation_style.csl | sed 's/<div class="references">//' | sed 's/<\/div>//' | pandoc -f html -t markdown | sed -e :a -e 's/<[^>]*>//g;/</N;//ba' > README.md
-->
JuliaNet
========

A deep neural network library for Julia. Features include:

-   Multi-classification
-   Momentum
-   Nesterov's Accelerated Gradient method [3]
-   Online, mini-batch, and batched learning
-   Custom stop criterion functions
-   Custom hyper paramater update criterion
-   Custom activation functions
-   Custom connections between layers
-   Random weight initializing based on fan-in/out heuristics [1]
-   Dropout [2]
-   Parallelization support

The library is written such that it is easy to make modifications and customizations to the network structure and optimization strategies. In fact, it is encouraged you do make modifications to suit your needs instead of exclusively treating it as a black box.


Getting started
---------------

See the examples directory for use cases.


Requirements
------------

-   Julia 0.3 or above


Roadmap
-------

Near-future releases want to feature:

-   L1/L2 weight penalties
-   Max incoming weight constraints and scaling
-   Convolutional and pooling layers
-   More test coverage

See the Issues tab for a full list of planned features.


References
----------
