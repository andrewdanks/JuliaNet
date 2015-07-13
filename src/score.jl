function predict(nn::NeuralNetwork, X::InputTensor, t::Vector{T_FLOAT})
    Y_prob = forward_pass(nn, X)
    predict(nn, Y_prob)
end

function predict(nn::NeuralNetwork, X::T_TENSOR, t::Vector{T_FLOAT})
    Y_prob = forward_pass(nn, InputTensor(X))
    predict(nn, Y_prob)
end

function predict(nn::NeuralNetwork, Y_prob::Matrix{T_FLOAT})
    cases = num_data_matrix_inputs(Y_prob)
    predictions = zeros(cases)
    for i=1:cases
        y_prob = Y_prob[:,i]
        class = get_class_from_prob(nn, y_prob)
        predictions[i] = class
    end
    predictions
end

function test_error(predictions::Vector{T_FLOAT}, targets::Vector)
    count(i -> i != 0, predictions - targets) / length(targets)
end

function get_class_from_prob(nn::NeuralNetwork, y_prob::Vector)
    _, class_idx = findmax(y_prob)
    nn.classes[class_idx]
end
