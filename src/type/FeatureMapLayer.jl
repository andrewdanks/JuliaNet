abstract FeatureMapLayer <: NeuralLayer


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
