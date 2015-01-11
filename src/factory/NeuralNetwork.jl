function SimpleFullyConnectedNeuralNetwork(
    num_features::T_INT,
    num_classes::T_INT,
    hidden_layer_sizes::Vector{T_INT}
    ;
    activator::Activator=SIGMOID_ACTIVATOR,
    weight_sampler::Function=((rows, cols) -> rand_range(-1., 1., rows, cols))
)
    input_layer = InputLayer(num_features)
    layer_sizes = vcat(num_features, hidden_layer_sizes, num_classes)
    hidden_layers = FullyConnectedHiddenLayers(layer_sizes,
        weight_sampler,
        activator=activator
    )
    output_layer = OutputLayer(ones(layer_sizes[end], num_classes), weight_sampler)
    NeuralNetwork(input_layer, hidden_layers, output_layer)
end