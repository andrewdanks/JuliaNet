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
    weight_sampler::Function=default_weight_sampler
)
    NeuralNetwork([
        HiddenLayer(activator, weight_sampler(input_size, hidden_layer_size))
        HiddenLayer(activator, weight_sampler(hidden_layer_size, input_size))
    ])
end


function Base.push!(nn::NeuralNetwork, layer::NeuralLayer)
    push!(nn.layers, layer)
end


function Base.push!(nn::NeuralNetwork, layer::ConvolutionalLayer)
    push!(nn.layers, layer)
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
