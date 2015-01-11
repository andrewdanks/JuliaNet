function L1(nn::NeuralNetwork)
    sum(abs(nn.layers[end].weights))
end

function L2(nn::NeuralNetwork)
    sum(nn.layers[end].weights .^ 2)
end