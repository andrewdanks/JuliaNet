type NeuralNetwork
    layers::Vector{NeuralLayer}
    classes::Vector

    NeuralNetwork() = new(NeuralLayer[])
    NeuralNetwork(layers::Vector{NeuralLayer}) = new(layers)
end

function Base.push!(nn::NeuralNetwork, layer_factory::Union(DataType, Function), args...)
    push!(nn.layers, layer_factory(nn.layers[end], args...))
end

function layer_at_index(nn::NeuralNetwork, i::T_INT)
    nn.layers[i]
end

function all_layers(nn::NeuralNetwork)
    nn.layers
end

function num_layers(nn::NeuralNetwork)
    length(all_layers(nn))
end

function input_layer(nn::NeuralNetwork)
    nn.layers[1]
end

function output_layer(nn::NeuralNetwork)
    nn.layers[end]
end

function update_weights!(nn::NeuralNetwork, params::HyperParams)
    for layer in nn.layers
        update_weights!(layer, params)
    end
end

function deactivate_network!(nn::NeuralNetwork)
    for layer in all_layers(nn)
        deactivate_layer!(layer)
    end
end

function verify_network(nn::NeuralNetwork)
    # Go through each layer and verify that the connections and
    # dimensions are correct. Ideally, if this check passes,
    # then we should never get errors such as dimension mismatch

    # TODO: implement

    # 1. ensure all connections are valid (appropriate weight matrix dims)
    # 2. ensure conv layers are in appropriate locations in the network?
    # 3. ensure pooling layer has same # of maps as previous conv layer

    true
end

function Base.show(io::IO, net::NeuralNetwork)
    println(io, "NeuralNetwork[", string(all_layers(net)), "]")
end