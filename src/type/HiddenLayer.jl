type HiddenLayer <: NeuralLayer
    # Activator is responsible for doing any transformations
    # to the input before it goes to the next layer
    activator::Activator

    # Dimensions: <pevious layer size x this layer size>
    # Each row corresponds to a neuron, so it represents the incoming
    # weights to that neuron. I.e., the value of (i,j) is the weight
    # from neuron i in the previous layer to neuron j in this layer
    weights::Matrix{T_FLOAT}

    # Dimensions: <this layer size>
    # Each component i represents the bias for the i'th neuron
    # biases::Vector{T_FLOAT}

    # undefined properties by default
    dropout_coefficient::T_FLOAT

    HiddenLayer(activator::Activator, weights::Matrix{T_FLOAT}) = new(activator, weights)

    function HiddenLayer(
        activator::Activator,

        # Dimensions: <pevious layer size x this layer size>
        # Matrix of 1s and 0s where 1s indicate where a connection is present
        # (i,j) == 1 indicate that neuron i in previous layer connects to
        # neuron j in this layer
        connections::Matrix{T_INT},
        
        # A function that can take a variable number of integer arguments
        # representing the dimensions of a vector or matrix that should
        # be returned with samples from some distribution
        weight_sampler::Function
    )
        prev_layer_size, this_layer_size = size(connections)

        weights = weight_sampler(prev_layer_size, this_layer_size)
        # Zero-out the spots where there are no connections
        weights = weights .* connections

        new(activator, weights)
    end

end


function input_size(layer::HiddenLayer)
    size(layer.weights)[1]
end


function Base.size(layer::HiddenLayer)
    size(layer.weights)[2]
end
