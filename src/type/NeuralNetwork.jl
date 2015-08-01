immutable type NeuralNetwork
    layers::Vector{NeuralLayer}
    classes::Vector

    NeuralNetwork() = new(NeuralLayer[])
    NeuralNetwork{T<:NeuralLayer}(layers::Vector{T}) = new(layers)
    NeuralNetwork{T<:NeuralLayer}(layers::Vector{T}, classes::Vector) = new(layers, classes)
end


function Autoencoder(
    input_size::T_UINT,
    hidden_layer_size::T_UINT,
    activator::Activator,
    corruption_level::T_FLOAT=0.0,
    weight_sampler::Function=default_weight_sampler
)
    StackedAutoencoder(input_size, [hidden_layer_size], activator, [corruption_level], weight_sampler)
end


function StackedAutoencoder(
    input_size::T_UINT,
    hidden_layer_sizes::Vector{T_UINT},
    activator::Activator,
    corruption_levels::Vector{T_FLOAT}=zeros(length(hidden_layer_sizes)),
    weight_sampler::Function=default_weight_sampler
)
    layers = HiddenLayer[]
    for (idx, layer_size) in enumerate(hidden_layer_sizes)
        corruption_level = corruption_levels[idx]
        first_layer = HiddenLayer(activator, weight_sampler(input_size, layer_size))
        first_layer.corruption_level = corruption_level
        push!(layers, first_layer)
        push!(layers, HiddenLayer(activator, weight_sampler(layer_size, input_size)))
    end

    NeuralNetwork(layers)
end


function Base.size(nn::NeuralNetwork)
    length(nn.layers)
end


function input_layer(nn::NeuralNetwork)
    nn.layers[1]
end


function output_layer(nn::NeuralNetwork)
    nn.layers[end]
end
