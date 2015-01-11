type InputLayer <: NeuralLayer
    size::T_INT
    num_maps::T_INT
    feature_map_size::T_2D
    activator::Activator

    activation::InputTensor

    InputLayer(
        num_features::T_INT,
        activator::Activator=IDENTITY_ACTIVATOR
    ) = InputLayer((num_features, 1), activator)
    
    InputLayer(
        feature_map_size::T_2D,
        activator::Activator=IDENTITY_ACTIVATOR
    ) = InputLayer(1, feature_map_size, activator)

    InputLayer(
        num_maps::T_INT,
        feature_map_size::T_2D,
        activator::Activator=IDENTITY_ACTIVATOR
    ) = new(num_maps * prod(feature_map_size), num_maps, feature_map_size, activator)
end

function Base.size(layer::InputLayer)
    layer.num_maps * prod(layer.feature_map_size)
end

function Base.show(io::IO, layer::InputLayer)
    layer_type = typeof(layer)
    properties = [size(layer), layer.num_maps, layer.feature_map_size]
    println(layer_type, "[", join(properties, "; "), "]")
end

function input(layer::InputLayer)
    layer.activation
end

function activate_layer!(
    layer::InputLayer,
    input::InputTensor,
    activation_fn::Function
)
    layer.activation = InputTensor(activation_fn(input.data))
end

function set_weight_gradients!(
    layer::InputLayer,
    params::HyperParams,
    error_signal::T_TENSOR
)
    # No weights for pooling layer, so do nothing
end
function update_weights!(layer::InputLayer, params::HyperParams)
    # No weights for pooling layer, so do nothing
end

function deactivate_layer!(layer::InputLayer)
end