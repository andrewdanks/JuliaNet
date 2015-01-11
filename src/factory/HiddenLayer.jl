function FullyConnectedHiddenLayers(
    sizes::Vector{T_INT},
    weight_sampler::Function,
    activator::Activator=SIGMOID_ACTIVATOR
)
    prev_layer_size = sizes[1]
    FullyConnectedHiddenLayers(
        prev_layer_size,
        sizes,
        weight_sampler,
        activator=activator
    )
end

function FullyConnectedHiddenLayers(
    input_layer::NeuralLayer,
    sizes::Vector{T_INT},
    weight_sampler::Function,
    activator::Activator
)
    hidden_layers = NeuralLayer[]
    prev_layer_size = input_layer.size
    for layer_size in sizes
        connections = ones(prev_layer_size, layer_size)
        hidden_layer = HiddenLayer(input_layer, int(connections), activator, weight_sampler)
        push!(hidden_layers, hidden_layer)
        prev_layer_size = layer_size
        input_layer = hidden_layer
    end
    hidden_layers
end

function FullyConnectedHiddenAndOutputLayers(
    input_layer::NeuralLayer,
    sizes::Vector{T_INT},
    num_classes::T_INT,
    weight_sampler::Function,
    activator::Activator=SIGMOID_ACTIVATOR,
    output_layer_activator::Activator=SOFTMAX_ACTIVATOR
)
    hidden_layers = FullyConnectedHiddenLayers(
        input_layer,
        sizes,
        weight_sampler,
        activator
    )
    output_layer = FullyConnectedOutputLayer(
        hidden_layers[end],
        num_classes,
        weight_sampler,
        output_layer_activator
    )
    hidden_layers, output_layer
end
