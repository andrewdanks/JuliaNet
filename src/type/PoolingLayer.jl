type PoolingLayer <: FeatureMapLayer
    size::T_INT
    input_layer::NeuralLayer
    activator::Activator

    feature_map_size::T_2D
    kernel_size::T_2D
    input_map_size::T_2D
    num_maps::T_INT
    num_input_maps::T_INT

    pre_activation::T_4D_TENSOR
    activation::InputTensor

    max_masks::T_4D_TENSOR

    function PoolingLayer(
        input_layer::NeuralLayer,
        kernel_size::T_2D
    )
        num_input_maps = input_layer.num_maps
        input_map_size = input_layer.feature_map_size

        # TODO: we should not enforce non-overlapping kernels
        assert(input_map_size[1] % kernel_size[1] == 0)
        assert(input_map_size[2] % kernel_size[2] == 0)

        num_maps = num_input_maps
        map_size = (input_map_size[1] / kernel_size[1], input_map_size[2] / kernel_size[2])
        layer_size = prod(map_size) * num_maps

        new(
            layer_size,
            input_layer,
            IDENTITY_ACTIVATOR,

            map_size,
            kernel_size,
            input_map_size,
            num_maps,
            num_input_maps
        )
    end
end

function activate_layer!(
    layer::PoolingLayer,
    input_tensor::InputTensor,
    activation_fn::Function
)
    activation = zeros(
        input_tensor.batch_size,
        layer.num_maps,
        layer.feature_map_size[1],
        layer.feature_map_size[2],
    )

    max_masks = zeros(
        input_tensor.batch_size,
        layer.num_maps,
        layer.feature_map_size[1] * layer.kernel_size[1],
        layer.feature_map_size[2] * layer.kernel_size[2]
    )

    for input_idx = 1:input_tensor.batch_size
        for map_idx = 1:layer.num_maps
            input_map = input_map_data(input_tensor, input_idx, map_idx)
            for p = 1:layer.feature_map_size[1]
                kernel_range1 = non_overlapping_receptive_field_range(layer.kernel_size[1], p)
                for q = 1:layer.feature_map_size[2]
                    kernel = non_overlapping_receptive_field(layer, (p, q), input_map)
                    kernel_range2 = non_overlapping_receptive_field_range(layer.kernel_size[2], q)
                    
                    max_val, max_idx = findmax(kernel)
                    max_mask = zeros(layer.kernel_size)
                    max_mask[max_idx] = 1

                    max_masks[input_idx, map_idx, kernel_range1, kernel_range2] = max_mask
                    activation[input_idx, map_idx, p, q] = max_val
                end
            end
        end
    end

    layer.max_masks = max_masks
    layer.pre_activation = activation
    layer.activation = InputTensor(activation_fn(activation))
end

function upsample(
    layer::PoolingLayer,
    x::T_2D_TENSOR
)
    kron(x, ones(layer.kernel_size))
end

function grad_error_wrt_net(
    layer::PoolingLayer,
    error_signal::T_4D_TENSOR
)    
    next_layer = layer.input_layer

    size_error_signal = size(error_signal)
    batch_size = size_error_signal[1]

    # Technically we should be able to use the same size as the error signal here
    next_error_signal = zeros(
        batch_size,
        next_layer.num_maps,
        next_layer.feature_map_size[1],
        next_layer.feature_map_size[2]
    )

    layer_grad_activation = layer.activator.grad_activation_fn
    grad_activation_fn = next_layer.activator.grad_activation_fn

    for x = 1:batch_size
        for i = 1:layer.num_maps
            map_error = squeeze(error_signal[x, i, :, :], (1, 2))
            # This is a hack that should be done in grad_error_wrt_net in ConvolutonalLayer, 
            # but is more efficient here since we're already iterating over the batch
            #map_error = map_error .* grad_activation_fn(squeeze(layer.pre_activation[x, i, :, :], (1, 2)))
            max_mask = squeeze(layer.max_masks[x, i, :, :], (1, 2))
            grad_pre_activation = squeeze(grad_activation_fn(next_layer.pre_activation[x, i, :, :]), (1, 2))
            next_error_signal[x, i, :, :] = max_mask .* grad_pre_activation .* upsample(layer, map_error)
        end
    end

    next_error_signal
end

function set_weight_gradients!(
    layer::PoolingLayer,
    error_signal::T_TENSOR
)
    # No weights for pooling layer, so do nothing
end
function update_weights!(layer::PoolingLayer, params::HyperParams)
    # No weights for pooling layer, so do nothing
end