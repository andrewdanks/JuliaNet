type NeuralNetwork
    layers::Vector{NeuralLayer}
    classes::Vector{Number}

    NeuralNetwork() = new(NeuralLayer[])
    NeuralNetwork{T<:NeuralLayer}(layers::Vector{T}) = new(layers)
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
