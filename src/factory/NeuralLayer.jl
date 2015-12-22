function FullyConnectedHiddenLayers(
    input_size::T_INT,
    sizes::Vector{T_INT},
    activator::Symbol,
    weight_sampler::Function=default_weight_sampler
)
    hidden_layers = NeuralLayer[]
    prev_layer_size = input_size
    for layer_size in sizes
        connections = ones(prev_layer_size, layer_size)
        hidden_layer = HiddenLayer(activator, to_int(connections), weight_sampler)
        push!(hidden_layers, hidden_layer)
        prev_layer_size = layer_size
        input_layer = hidden_layer
    end
    hidden_layers
end


function FullyConnectedHiddenAndOutputLayers(
    input_size::T_INT,
    sizes::Vector{T_INT},
    num_classes::T_INT,
    activator::Symbol=:sigmoid,
    output_layer_activator::Symbol=:softmax,
    weight_sampler::Function=default_weight_sampler
)
    hidden_layers = FullyConnectedHiddenLayers(
        input_size, sizes, activator, weight_sampler
    )
    output_layer = FullyConnectedOutputLayer(
        sizes[end], num_classes, output_layer_activator, weight_sampler
    )
    hidden_layers, output_layer
end


function FullyConnectedOutputLayer(
    input_size::T_UINT,
    num_classes::T_UINT,
    activator::Symbol=:softmax,
    weight_sampler::Function=default_weight_sampler
)
    HiddenLayer(
        activator,
        to_int(ones(input_size, num_classes)),
        weight_sampler
    )
end
