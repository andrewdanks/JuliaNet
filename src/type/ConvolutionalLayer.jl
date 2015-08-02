type ConvolutionalLayer <: FeatureMapLayer
    activator::Symbol

    weights::T_4D_TENSOR

    feature_map_size::T_2D
    kernel_size::T_2D
    num_maps::T_UINT
    num_input_maps::T_UINT

    function ConvolutionalLayer(
        num_input_maps::T_UINT,
        input_map_size::T_2D,
        kernel_size::T_2D,
        num_maps::T_INT,
        activator::Symbol,
        weight_sampler::Function=default_weight_sampler
    )
        map_size = get_feature_map_size(input_map_size, kernel_size)

        weights = make_weights(
            (num_input_maps, num_maps, kernel_size[1], kernel_size[2]),
            default_weight_sampler
        )

        new(
            activator,
            weights,
            map_size,
            kernel_size,
            num_maps,
            num_input_maps
        )
    end
end


function get_feature_map_size(input_map_size::T_2D, kernel_size::T_2D)
    (input_map_size[1] - kernel_size[1] + 1, input_map_size[2] - kernel_size[2] + 1)
end


function get_kernel_weights(layer::ConvolutionalLayer, input_map_idx::T_INT, map_idx::T_INT)
    @inbounds return squeeze(layer.weights[input_map_idx, map_idx, :, :], (1, 2))
end


function get_kernel_weights_3d(weights::T_2D_TENSOR)
    size_weights = size(weights)
    @inbounds return reshape(weights, 1, size_weights[1], size_weights[2])
end


function get_pre_activation(layer::ConvolutionalLayer, input::InputTensor)
    pre_activation = zeros(
        input.batch_size,
        layer.num_maps,
        layer.feature_map_size[1],
        layer.feature_map_size[2],
    )
    
    for map_idx = 1:layer.num_maps
        map_pre_activation = zeros(
            input.batch_size,
            layer.feature_map_size[1],
            layer.feature_map_size[2]
        )
        for input_map_idx = 1:layer.num_input_maps
            input_map = map_data(input, input_map_idx)
            weights = get_kernel_weights(layer, input_map_idx, map_idx)
            map_pre_activation += convn_valid(input_map, get_kernel_weights_3d(weights))
        end
        @inbounds pre_activation[:, map_idx, :, :] = map_pre_activation
    end
    pre_activation
end


function get_grad_weights(
    layer::LinkedLayer{ConvolutionalLayer},
    error_signal::T_4D_TENSOR
)
    size_error_signal = size(error_signal)
    batch_size = size_error_signal[1]

    grad_weights = zeros(
        layer.data_layer.num_input_maps, layer.data_layer.num_maps,
        layer.data_layer.kernel_size[1], layer.data_layer.kernel_size[2]
    )

    for i = 1:layer.data_layer.num_input_maps
        for j = 1:layer.data_layer.num_maps
            map_error = squeeze(error_signal[:, j, :, :], 2)
            map_input = map_data(layer.input, i)
            rotated_map_input = flipall(map_input)

            grad_map_weights = flipall(convn_valid(rotated_map_input, map_error))
            @inbounds grad_weights[i, j, :, :] = reshape(grad_map_weights, layer.data_layer.kernel_size)
        end
    end

    layer.grad_weights = grad_weights / batch_size
end


function get_grad_error_wrt_net{T<:ConvolutionalLayer}(
    layer::LinkedLayer{T},
    error_signal::T_4D_TENSOR
)
    size_error_signal = size(error_signal)
    batch_size = size_error_signal[1]

    next_error_signal = zeros(
        batch_size,
        layer.prev.data_layer.num_maps,
        layer.prev.data_layer.feature_map_size[1],
        layer.prev.data_layer.feature_map_size[2]
    )

    for i = 1:layer.prev.data_layer.num_maps
        next_map_error = zeros(
            batch_size,
            layer.prev.data_layer.feature_map_size[1],
            layer.prev.data_layer.feature_map_size[2]
        )
        for j = 1:layer.data_layer.num_maps
            map_error = squeeze(error_signal[:, j, :, :], 2)
            weights = flipall(get_kernel_weights(layer, i, j))
            next_map_error += convn_full(map_error, get_kernel_weights_3d(weights))
        end
        next_error_signal[:, i, :, :] = next_map_error
    end

    next_error_signal
end
