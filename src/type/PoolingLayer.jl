type PoolingLayer <: FeatureMapLayer
    activator::Activator

    feature_map_size::T_2D
    kernel_size::T_2D
    input_map_size::T_2D
    num_maps::T_UINT
    num_input_maps::T_UINT

    max_masks::T_TENSOR

    # undefined properties by default
    dropout_coefficient::T_FLOAT

    function PoolingLayer(
        num_input_maps::T_UINT,
        input_map_size::T_2D,
        kernel_size::T_2D
    )
        assert(input_map_size[1] % kernel_size[1] == 0)
        assert(input_map_size[2] % kernel_size[2] == 0)

        num_maps = num_input_maps
        map_size = (input_map_size[1] / kernel_size[1], input_map_size[2] / kernel_size[2])
        layer_size = prod(map_size) * num_maps

        new(
            IDENTITY_ACTIVATOR,
            map_size,
            kernel_size,
            input_map_size,
            num_maps,
            num_input_maps
        )
    end
end


function update_weights!(
    linked_layer::LinkedLayer{PoolingLayer},
    params::HyperParams
)
    # no-op
end


function get_grad_weights(
    layer::LinkedLayer{PoolingLayer},
    error_signal::T_4D_TENSOR
)
    zeros(1,1)
end 


function get_pre_activation(
    layer::PoolingLayer,
    input_tensor::InputTensor
)
    pre_activation = zeros(
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
                kernel_range1 = non_overlapping_receptive_field_range(
                    layer.kernel_size[1], p
                )
                for q = 1:layer.feature_map_size[2]
                    kernel = non_overlapping_receptive_field(layer, (p, q), input_map)
                    kernel_range2 = non_overlapping_receptive_field_range(
                        layer.kernel_size[2], q
                    )
                    
                    max_val, max_idx = findmax(kernel)
                    max_mask = zeros(layer.kernel_size)
                    max_mask[max_idx] = 1

                    max_masks[input_idx, map_idx, kernel_range1, kernel_range2] = max_mask
                    pre_activation[input_idx, map_idx, p, q] = max_val
                end
            end
        end
    end

    # obvious hack is obvious
    layer.max_masks = max_masks

    pre_activation
end


function get_grad_error_wrt_net(
    layer::LinkedLayer{PoolingLayer},
    error_signal::T_4D_TENSOR
)    
    function upsample(
        layer::PoolingLayer,
        x::T_2D_TENSOR
    )
        kron(x, ones(layer.kernel_size))
    end

    size_error_signal = size(error_signal)
    batch_size = size_error_signal[1]

    # Technically we should be able to use the same size as the error signal here
    next_error_signal = zeros(
        batch_size,
        layer.prev.data_layer.num_maps,
        layer.prev.data_layer.feature_map_size[1],
        layer.prev.data_layer.feature_map_size[2]
    )

    layer_grad_activation = layer.data_layer.activator.grad_activation_fn
    grad_activation_fn = layer.prev.data_layer.activator.grad_activation_fn

    for x = 1:batch_size
        for i = 1:layer.data_layer.num_maps
            map_error = squeeze(error_signal[x, i, :, :], (1, 2))
            # This is a hack that should be done in grad_error_wrt_net in ConvolutonalLayer, 
            # but is more efficient here since we're already iterating over the batch
            #map_error = map_error .* grad_activation_fn(squeeze(layer.pre_activation[x, i, :, :], (1, 2)))
            max_mask = squeeze(layer.data_layer.max_masks[x, i, :, :], (1, 2))
            grad_pre_activation = squeeze(grad_activation_fn(
                layer.prev.pre_activation[x, i, :, :]
            ), (1, 2))
            next_error_signal[x, i, :, :] = max_mask .* grad_pre_activation .* upsample(layer.data_layer, map_error)
        end
    end

    next_error_signal
end


function receptive_field(
    feature_map::Matrix{T_FLOAT},
    kernel_size::T_2D,
    i::T_INT,
    j::T_INT
)
    range1 = receptive_field_range(kernel_size[1], i)
    range2 = receptive_field_range(kernel_size[2], j)
    feature_map[range1, range2]
end



function receptive_field_range(kernel_size::T_INT, i::T_INT)
     i : (i + kernel_size - 1)
end


function non_overlapping_receptive_field(
    layer::FeatureMapLayer,
    kernel_idxs::T_2D,
    input_map::T_2D_TENSOR
)
    range1 = non_overlapping_receptive_field_range(layer.kernel_size[1], kernel_idxs[1])
    range2 = non_overlapping_receptive_field_range(layer.kernel_size[2], kernel_idxs[2])
    
    input_map[range1, range2]
end


function non_overlapping_receptive_field_range(
    kernel_size::T_INT,
    kernel_idx::T_INT,
)
    start = (kernel_idx - 1) * kernel_size + 1
    stop = kernel_idx * kernel_size

    start:stop
end
