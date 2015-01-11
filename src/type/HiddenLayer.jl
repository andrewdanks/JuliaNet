type HiddenLayer <: NeuralLayer
    size::T_INT

    input_layer::NeuralLayer

    # Activator is responsible for doing any transformations
    # to the input before it goes to the next layer
    activator::Activator

    connections::Matrix

    # Dimensions: <pevious layer size x this layer size>
    # Each row corresponds to a neuron, so it represents the incoming
    # weights to that neuron. I.e., the value of (i,j) is the weight
    # from neuron i in the previous layer to neuron j in this layer
    weights::Matrix{T_FLOAT}

    # Dimensions: <this layer size>
    # Each component i represents the bias for the i'th neuron
    biases::Vector{T_FLOAT}

    # Properties that only pertain to the layer while training

    prev_weight_delta::Matrix{T_FLOAT}
    prev_bias_delta::Vector{T_FLOAT}

    pre_activation::Matrix{T_FLOAT}
    activation::InputTensor
    grad_weights::Matrix{T_FLOAT}

    function HiddenLayer(
        input_layer::NeuralLayer,

        # Dimensions: <pevious layer size x this layer size>
        # Matrix of 1s and 0s where 1s indicate where a connection is present
        # (i,j) == 1 indicate that neuron i in previous layer connects to
        # neuron j in this layer
        connections::Matrix{T_INT},

        activator::Activator,
        
        # A function that can take a variable number of integer arguments
        # representing the dimensions of a vector or matrix that should
        # be returned with samples from some distribution
        weight_sampler::Function
    )

        prev_layer_size, this_layer_size = size(connections)

        weights = weight_sampler(prev_layer_size, this_layer_size)
        # Zero-out the spots where there are no connections
        weights = weights .* connections

        biases = vectorize(zeros(this_layer_size, 1))

        prev_weight_delta = zeros(size(weights))
        prev_bias_delta = zeros(size(biases))

        new(
            this_layer_size,
            input_layer,
            activator,
            connections,
            weights,
            biases,
            prev_weight_delta,
            prev_bias_delta
        )
    end

end

function activate_layer!(
    layer::HiddenLayer,
    input::InputTensor,
    activation_fn::Function
)
    vectorized_input = vectorized_data(input)    
    layer.pre_activation = layer.weights' * vectorized_input
    layer.activation = InputTensor(activation_fn(layer.pre_activation))
end

function set_weight_gradients!(layer::HiddenLayer, error_signal::T_TENSOR)
    input = vectorized_data(layer_input(layer))
    batch_size = size(error_signal)[2]
    grad_weights = (input * error_signal') / batch_size
    layer.grad_weights = layer.connections .* grad_weights
end

function Base.size(layer::HiddenLayer)
    layer.size
end

function Base.show(io::IO, layer::HiddenLayer)
    layer_type = typeof(layer)
    println(layer_type, "[", layer.size, "; ", layer.activator, "]")
end