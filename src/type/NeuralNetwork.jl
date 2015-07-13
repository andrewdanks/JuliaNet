type NeuralNetwork
    layers::Vector{NeuralLayer}
    classes::Vector

    NeuralNetwork() = new(NeuralLayer[])
    NeuralNetwork(layers::Vector{NeuralLayer}) = new(layers)
end


function Base.push!(nn::NeuralNetwork, layer::NeuralLayer)
    # TODO: check that there's an input layer
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
