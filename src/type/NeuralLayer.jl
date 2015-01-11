abstract NeuralLayer

function activate_layer!(layer::NeuralLayer, input::InputTensor)
    activate_layer!(layer, input, layer.activator.activation_fn)
end

function test_activate_layer!(layer::NeuralLayer, input::InputTensor)
    activate_layer!(layer, input, layer.activator.test_activation_fn)
end

function backward_pass!(
    layer::NeuralLayer,
    params::HyperParams,
    error_signal::T_TENSOR,
    next_layer
)
    set_weight_gradients!(layer, error_signal)
    
    if next_layer != None
        next_error_signal = grad_error_wrt_net(layer, error_signal)
        return format_error_signal(next_layer, next_error_signal)
    end
end

function update_weights!(layer::NeuralLayer, params::HyperParams)
    learning_rate = params.learning_rate
    momentum = params.momentum

    weight_delta = learning_rate * layer.grad_weights
    if momentum > 0
        layer.prev_weight_delta = momentum * layer.prev_weight_delta + weight_delta
        weight_delta = layer.prev_weight_delta
    end

    layer.weights -= weight_delta
end

function grad_error_wrt_net(
    layer::NeuralLayer,
    error_signal::T_2D_TENSOR
)
    next_layer = layer.input_layer
    # Dimensions of next error signal should be <layer fan in> x <batch size>
    grad_activation_fn = next_layer.activator.grad_activation_fn

    # TODO: find a non-hacky way to do this
    next_layer_pre_activation = vectorized_data(InputTensor(next_layer.pre_activation))

    (layer.weights * error_signal) .* grad_activation_fn(next_layer_pre_activation)
end

function format_error_signal(
    layer::NeuralLayer,
    error_signal::T_TENSOR
)
    # TODO: perhaps Error Signal should also be an InputTensor
    # so we can remove this extra logic

    error_signal_size = size(error_signal)
    error_signal_dims = length(error_signal_size)

    # if error_signal_dims == 4
    #     batch_size = error_signal_size[1]
    #     new_error_signal = zeros(prod(error_signal_size[2:end]), batch_size)
    #     for x = 1:batch_size
    #         new_error_signal[:, x] = vec(error_signal[x, :, :, :])
    #     end
    #     return new_error_signal
    # end

    error_signal
end

function layer_input(layer::NeuralLayer)
    layer.input_layer.activation
end

function deactivate_layer!(layer::NeuralLayer)
    # layer.input = 0.
    # layer.pre_activation = 0.
    # layer.activation = 0.
    # layer.prev_weight_delta = 0.
    # layer.prev_bias_delta = 0.
    # gc()
end