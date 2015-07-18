using JuliaNet
using MNIST

X, Y = MNIST.traindata()
trainX, trainY = X[:, 1:10000], Y[1:10000]
validX, validY = X[:, 50001:end], Y[50001:end]
testX, testY = MNIST.testdata()

prcoessed_trainX = unit_variance(zero_mean(trainX))
processed_validX = unit_variance(zero_mean(validX))
processed_textX = unit_variance(zero_mean(testX))

function sample_weights(rows, cols=1)
    val = 4 * sqrt(6 / (rows + cols))
    rand_range(-val, val, rows, cols)
end

function sample_weights(dims::(Number, Number))
    sample_weights(dims[1], dims[2])
end

num_classes = 10
input_maps = 1
input_map_size = (28, 28)

srand(123)

conv_layer1 = ConvolutionalLayer(
    input_maps, input_map_size,
    (5, 5), 4,
    TANH_ACTIVATOR,
    sample_weights
)

pooling_layer1 = PoolingLayer(
    conv_layer1.num_maps,
    conv_layer1.feature_map_size,
    (2, 2)
)

cnn_layers = vcat(conv_layer1, pooling_layer1)


hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
    size(cnn_layers[end]),
    [800, 800],
    num_classes,
    sample_weights,
    SIGMOID_ACTIVATOR
)

net = NeuralNetwork(vcat(cnn_layers, hidden_layers, output_layer))

params = HyperParams()
params.batch_size = 100
params.epochs = 5
params.momentum = 0.4
params.learning_rate = 0.7

fit!(
    net,
    prcoessed_trainX,
    trainY,
    params,
    valid_data=processed_validX,
    valid_targets=validY,
    input_map_size=input_map_size
)

predictions = predict(net, InputTensor(processed_textX, input_map_size), testY)
println("test error: ", test_error(predictions, testY))
