using JuliaNet
using MNIST

srand(34455)

X, Y = MNIST.traindata()
trainX, trainY = X[:, 1:50000], Y[1:50000]
validX, validY = X[:, 50001:end], Y[50001:end]
testX, testY = MNIST.testdata()

trainX = unit_variance(zero_mean(trainX))
validX = unit_variance(zero_mean(validX))
textX = unit_variance(zero_mean(testX))

input_maps = 1
input_map_size = (28, 28)
classes = [0,1,2,3,4,5,6,7,8,9]
num_classes = length(classes)

conv_layer = ConvolutionalLayer(
    input_maps, input_map_size,
    (5, 5), 4,
    :sigmoid
)

pooling_layer = PoolingLayer(
    conv_layer.num_maps,
    conv_layer.feature_map_size,
    (2, 2)
)

cnn_layers = vcat(conv_layer, pooling_layer)

hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
    size(cnn_layers[end]), [1200, 1200], num_classes, :sigmoid
)
hidden_layers[1].dropout_coefficient = 0.5
hidden_layers[2].dropout_coefficient = 0.5

nn = NeuralNetwork(vcat(cnn_layers, hidden_layers, output_layer), classes)
batches = make_batches(trainX, 100, classes, trainY, input_map_size)
validation_batch = make_batch(validX, classes, validY, input_map_size)

params = HyperParams()
params.epochs = 10
params.momentum = 0.4
params.learning_rate = 0.7

fit!(
    nn,
    params,
    batches,
    validation_batch,
    parallelize=false
)

predictions = predict(nn, textX, testY)
println("test error: ", test_error(predictions, testY))
