abstract NeuralLayer

function format_error_signal(layer::NeuralLayer, error_signal::T_TENSOR)
    size_error_signal = size(error_signal)
    error_signal_dims = length(size_error_signal)

    if error_signal_dims == 4 && length(size(layer.prev.data_layer.weights)) == 2
        vectorized_data(InputTensor(error_signal))
    else
        error_signal
    end
end
