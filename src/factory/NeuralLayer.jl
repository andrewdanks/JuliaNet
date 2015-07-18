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
    input_size::T_INT,
    sizes::Vector{T_INT},
    weight_sampler::Function,
    activator::Activator
)
    hidden_layers = NeuralLayer[]
    prev_layer_size = input_size
    for layer_size in sizes
        connections = ones(prev_layer_size, layer_size)
        hidden_layer = HiddenLayer(int(connections), activator, weight_sampler)
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
    weight_sampler::Function,
    activator::Activator=SIGMOID_ACTIVATOR,
    output_layer_activator::Activator=SOFTMAX_ACTIVATOR
)
    hidden_layers = FullyConnectedHiddenLayers(
        input_size,
        sizes,
        weight_sampler,
        activator
    )
    output_layer = FullyConnectedOutputLayer(
        sizes[end],
        num_classes,
        weight_sampler,
        output_layer_activator
    )
    hidden_layers, output_layer
end


function FullyConnectedOutputLayer(
    input_size::T_UINT,
    num_classes::T_UINT,
    weight_sampler::Function,
    activator::Activator=SOFTMAX_ACTIVATOR
)
    HiddenLayer(
        int(ones(input_size, num_classes)),
        activator,
        weight_sampler
    )
end
