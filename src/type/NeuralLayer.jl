abstract NeuralLayer


function update_weights!{T<:NeuralLayer}(layer::LinkedLayer{T}, params::HyperParams)
    prev_weight_delta = 0
    if params.momentum > 0 && isdefined(layer, :prev_weight_delta)
        prev_weight_delta = layer.prev_weight_delta 
    end
    weight_delta = params.momentum * prev_weight_delta - params.learning_rate * layer.grad_weights
    layer.weight_delta = weight_delta
    layer.data_layer.weights += weight_delta
end


function get_pre_activation(layer::NeuralLayer, input::InputTensor)
    layer.weights' * vectorized_data(input)
end


function get_grad_weights{T<:NeuralLayer}(
    layer::LinkedLayer{T},
    error_signal::T_2D_TENSOR
)
    batch_size = size(error_signal)[2]
    vectorized_data(layer.input) * error_signal' / batch_size
end


function get_grad_error_wrt_net{T<:NeuralLayer}(
    layer::LinkedLayer{T},
    error_signal::T_2D_TENSOR
)
    grad_activation_fn = layer.prev.data_layer.activator.grad_activation_fn
    weights = layer.data_layer.weights
    prev_pre_activation = vectorized_data(InputTensor(layer.prev.pre_activation))
    (weights * error_signal) .* grad_activation_fn(prev_pre_activation)
end


function format_error_signal(layer::NeuralLayer, error_signal::T_TENSOR)
    size_error_signal = size(error_signal)
    error_signal_dims = length(size_error_signal)

    if error_signal_dims == 4 && length(size(layer.prev.data_layer.weights)) == 2
        vectorized_data(InputTensor(error_signal))
    else
        error_signal
    end
end
