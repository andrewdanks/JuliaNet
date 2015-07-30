using JuliaNet
using MNIST

X, Y = MNIST.traindata()
trainX, trainY = X[:, 1:5000], Y[1:5000]
validX, validY = X[:, 50001:end], Y[50001:end]
testX, testY = MNIST.testdata()

trainX = unit_variance(zero_mean(trainX))
validX = unit_variance(zero_mean(validX))
textX = unit_variance(zero_mean(testX))

num_features = 28 * 28
num_classes = 10

function sample_weights(rows, cols=1)
    val = 4 * sqrt(6 / (rows + cols))
    rand_range(-val, val, rows, cols)
end

function sample_weights(dims::(Number, Number))
    sample_weights(dims[1], dims[2])
end


srand(34455)
hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
    num_features,
    [1200, 1200],
    num_classes,
    sample_weights,
    SIGMOID_ACTIVATOR
)
hidden_layers[1].dropout_coefficient = 0.5
hidden_layers[2].dropout_coefficient = 0.5

classes = [0,1,2,3,4,5,6,7,8,9]
nn = NeuralNetwork(vcat(hidden_layers, output_layer), classes)
batches = make_batches(trainX, 100, classes, trainY)
validation_batch = get_batch(validX, classes, validY)

params = HyperParams()
params.batch_size = 100
params.epochs = 5
params.momentum = 0.4
params.learning_rate = 0.7

fit!(
    nn,
    params,
    batches,
    validation_batch
)

predictions = predict(nn, processed_textX, testY)
println("test error: ", test_error(predictions, testY))
