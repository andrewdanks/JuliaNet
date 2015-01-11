type ConvolutionalLayer <: FeatureMapLayer
    size::T_INT
    input_layer::NeuralLayer
    activator::Activator
    weights::T_4D_TENSOR
    biases::T_1D_TENSOR

    feature_map_size::T_2D
    kernel_size::T_2D
    input_map_size::T_2D
    num_maps::T_INT
    num_input_maps::T_INT

    prev_weight_delta::T_4D_TENSOR
    prev_bias_delta::T_1D_TENSOR

    pre_activation::T_4D_TENSOR
    activation::InputTensor

    grad_weights::T_4D_TENSOR

    function ConvolutionalLayer(
        input_layer::NeuralLayer,
        kernel_size::T_2D,
        num_maps::T_INT,
        activator::Activator,
        weight_sampler::Function
    )
        num_input_maps = input_layer.num_maps
        input_map_size = input_layer.feature_map_size

        # An input to the network is generally:
        # <num features> x <num data vectors> x <num channels>
        # <batch size> x <num feature maps or channels> x <data cols> x <data rows>
        # Unless the data represents an image, for example, <data rows> will just be 1.

        # TODO: we should allow for more customization as to how many neurons are
        # in each feature map. I.e., maybe we don't want to gather input from every
        # possible receiptive field in the previous layer
        map_size = feature_map_size(input_map_size, kernel_size)
        layer_size = prod(map_size) * num_maps

        # All units in a feature map share the same set of weights
        weights = zeros(num_input_maps, num_maps, kernel_size[1], kernel_size[2])
        for i = 1:num_maps
            for j = 1:num_input_maps
                receptive_field_weights = weight_sampler(kernel_size[1], kernel_size[2])
                weights[j, i, :, :] = receptive_field_weights
            end
        end

        biases = zeros(num_maps)

        prev_weight_delta = zeros(size(weights))
        prev_bias_delta = zeros(size(biases))

        new(
            layer_size,
            input_layer,
            activator,
            weights,
            biases,

            map_size,
            kernel_size,
            input_map_size,
            num_maps,
            num_input_maps,

            prev_weight_delta,
            prev_bias_delta
        )
    end
end

function feature_map_size(input_map_size::T_2D, kernel_size::T_2D)
    (input_map_size[1] - kernel_size[1] + 1, input_map_size[2] - kernel_size[2] + 1)
end

function kernel_weights(layer::ConvolutionalLayer, input_map_idx::T_INT, map_idx::T_INT)
    @inbounds return squeeze(layer.weights[input_map_idx, map_idx, :, :], (1, 2))
end

function kernel_weights_3d(weights::T_2D_TENSOR)
    size_weights = size(weights)
    @inbounds return reshape(weights, 1, size_weights[1], size_weights[2])
end

function feature_map_preactivation(
    layer::ConvolutionalLayer,
    input_map::T_3D_TENSOR,
    input_map_idx::T_INT,
    map_idx::T_INT
)
    weights = kernel_weights(layer, input_map_idx, map_idx)
    bias = layer.biases[map_idx]
    pre_activation = convn_valid(input_map, kernel_weights_3d(weights)) .+ bias
    pre_activation
end

function activate_layer!(
    layer::ConvolutionalLayer,
    input_tensor::InputTensor,
    activation_fn::Function
)
    pre_activation = zeros(
        input_tensor.batch_size,
        layer.num_maps,
        layer.feature_map_size[1],
        layer.feature_map_size[2],
    )
    
    for map_idx = 1:layer.num_maps
        map_pre_activation = zeros(input_tensor.batch_size, layer.feature_map_size[1], layer.feature_map_size[2])
        for input_map_idx = 1:layer.num_input_maps
            input_map = map_data(input_tensor, input_map_idx)
            map_pre_activation += feature_map_preactivation(layer, input_map, input_map_idx, map_idx)
        end
        @inbounds pre_activation[:, map_idx, :, :] = map_pre_activation
    end

    layer.pre_activation = pre_activation
    layer.activation = InputTensor(activation_fn(pre_activation))
end

function set_weight_gradients!(layer::ConvolutionalLayer, error_signal::T_4D_TENSOR)
    size_error_signal = size(error_signal)
    batch_size = size_error_signal[1]

    grad_weights = zeros(layer.num_input_maps, layer.num_maps, layer.kernel_size[1], layer.kernel_size[2])

    for i = 1:layer.num_input_maps
        for j = 1:layer.num_maps
            map_error = squeeze(error_signal[:, j, :, :], 2)
            map_input = map_data(layer_input(layer), i)
            rotated_map_input = flipall(map_input)

            grad_map_weights = flipall(convn_valid(rotated_map_input, map_error))
            @inbounds grad_weights[i, j, :, :] = reshape(grad_map_weights, layer.kernel_size)
        end
    end

    layer.grad_weights = grad_weights / batch_size
end

# function grad_error_wrt_bias(layer::ConvolutionalLayer, error_signal::T_4D_TENSOR)
#     grad_biases = zeros(size(layer.biases))
#     for j = 1:layer.num_maps
#         grad_biases[j] = sum(error_signal[:, j, :, :])
#     end
#     grad_biases
# end

function grad_error_wrt_net(
    layer::ConvolutionalLayer,
    error_signal::T_4D_TENSOR
)
    next_layer = layer.input_layer

    size_error_signal = size(error_signal)
    batch_size = size_error_signal[1]

    next_error_signal = zeros(
        batch_size,
        next_layer.num_maps,
        next_layer.feature_map_size[1],
        next_layer.feature_map_size[2]
    )

    for i = 1:next_layer.num_maps
        next_map_error = zeros(batch_size, next_layer.feature_map_size[1], next_layer.feature_map_size[2])
        for j = 1:layer.num_maps
            map_error = squeeze(error_signal[:, j, :, :], 2)
            weights = flipall(kernel_weights(layer, i, j))
            next_map_error += convn_full(map_error, kernel_weights_3d(weights))
        end
        next_error_signal[:, i, :, :] = next_map_error
    end

    next_error_signal
end
