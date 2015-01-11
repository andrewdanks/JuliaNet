using JuliaNet
using MNIST

X, Y = MNIST.traindata()
trainX, trainY = X[:, 1:50000], Y[1:50000]
validX, validY = X[:, 50001:end], Y[50001:end]
testX, testY = MNIST.testdata()

prcoessed_trainX = unit_variance(zero_mean(trainX))
processed_validX = unit_variance(zero_mean(validX))
processed_textX = unit_variance(zero_mean(testX))

num_features = 28 * 28
num_classes = 10

function sample_weights(rows, cols=1)
    val = 4 * sqrt(6 / (rows + cols))
    rand_range(-val, val, rows, cols)
end

function sample_weights(dims::(Number, Number))
    sample_weights(dims[1], dims[2])
end

input_layer = InputLayer(num_features)

srand(34454)
hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
    input_layer,
    [800, 800],
    num_classes,
    sample_weights,
    SIGMOID_ACTIVATOR
)

net = NeuralNetwork(vcat(input_layer, hidden_layers, output_layer))

params = HyperParams()
params.batch_size = 50
params.epochs = 100
params.momentum = 0.7
params.learning_rate = 0.5

fit!(net, prcoessed_trainX, trainY, params,
    valid_data=processed_validX,
    valid_targets=validY,
)

predictions = predict(net, processed_textX, testY)
println("test accuracy: ", test_error(predictions, testY))
