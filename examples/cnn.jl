using JuliaNet
using MNIST

X, Y = MNIST.traindata()
trainX, trainY = X[:, 1:50000], Y[1:50000]
validX, validY = X[:, 50001:end], Y[50001:end]
testX, testY = MNIST.testdata()

prcoessed_trainX = unit_variance(zero_mean(trainX))
processed_validX = unit_variance(zero_mean(validX))
processed_textX = unit_variance(zero_mean(testX))

function my_weight_sampler(rows, cols=1)
    val = 4 * sqrt(6 / (rows + cols))
    rand_range(-val, val, rows, cols)
end

function my_weight_sampler(dims::(Number, Number))
    sample_weights(dims[1], dims[2])
end

num_classes = 10
input_dimensions = (28, 28)

input_layer = InputLayer(input_dimensions)

conv_layer1 = ConvolutionalLayer(
    input_layer,
    (5, 5), 4,
    TANH_ACTIVATOR,
    my_weight_sampler
)

pooling_layer1 = PoolingLayer(conv_layer1, (2, 2))

conv_layer2 = ConvolutionalLayer(
    pooling_layer1,
    (5, 5), 8,
    TANH_ACTIVATOR,
    my_weight_sampler
)

pooling_layer2 = PoolingLayer(conv_layer2, (2, 2))

cnn_layers = vcat(conv_layer1, pooling_layer1, conv_layer2, pooling_layer2)
output_layer = FullyConnectedOutputLayer(cnn_layers[end], num_classes, my_weight_sampler)

net = NeuralNetwork(vcat(input_layer, cnn_layers, output_layer))

params = HyperParams()
params.batch_size = 1
params.epochs = 100
params.momentum = 0.
params.learning_rate = 0.1

fit!(net, prcoessed_trainX, trainY, params, valid_data=processed_validX, valid_targets=validY)

predictions = predict(net, processed_textX, testY)
println("test accuracy: ", accuracy(predictions, testY))

