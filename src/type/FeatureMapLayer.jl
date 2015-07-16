abstract FeatureMapLayer <: NeuralLayer

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

function format_error_signal(layer::FeatureMapLayer, error_signal::T_TENSOR)
    size_error_signal = size(error_signal)
    error_signal_dims = length(size_error_signal)

    if error_signal_dims == 2
        batch_size = size_error_signal[2]
        
        new_error_signal = zeros(
            batch_size,
            layer.num_maps,
            layer.feature_map_size[1],
            layer.feature_map_size[1]
        )

        for x = 1:batch_size
            new_error_signal[x, :, :, :] = reshape(error_signal[:, x], layer.num_maps, layer.feature_map_size[1], layer.feature_map_size[2])
        end

        return new_error_signal
    end

    error_signal
end

function Base.size(layer::FeatureMapLayer)
    layer.num_maps * prod(layer.feature_map_size)
end
