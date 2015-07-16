function predict(nn::NeuralNetwork, X::InputTensor, t::Vector{T_FLOAT})
    predict(nn, forward_pass(nn, X))
end

function predict(nn::NeuralNetwork, X::T_TENSOR, t::Vector{T_FLOAT})
    predict(nn, InputTensor(X), t)
end

function predict(nn::NeuralNetwork, output::Matrix{T_FLOAT})
    cases = num_data_matrix_inputs(output)
    predictions = zeros(cases)
    for i=1:cases
        output_vec = output[:,i]
        predictions[i] = get_class_from_output(nn.classes, output_vec)
    end
    predictions
end

function test_error(predictions::Vector, targets::Vector)
    count(i -> i != 0, predictions - targets) / length(targets)
end

function get_class_from_output(classes::Vector, output::Vector)
    _, class_idx = findmax(output)
    classes[class_idx]
end
