using JuliaNet
using MNIST

srand(34455)

X, Y = MNIST.traindata()
trainX, trainY = X[:, 1:10000], Y[1:10000]
validX, validY = X[:, 50001:end], Y[50001:end]
testX, testY = MNIST.testdata()

trainX = unit_variance(zero_mean(trainX))
validX = unit_variance(zero_mean(validX))
textX = unit_variance(zero_mean(testX))

classes = [0,1,2,3,4,5,6,7,8,9]
num_features = 28 * 28
num_classes = length(classes)

hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
    num_features, [1200, 1200], num_classes, :sigmoid
)
hidden_layers[1].dropout_coefficient = 0.5
hidden_layers[2].dropout_coefficient = 0.5

nn = NeuralNetwork(vcat(hidden_layers, output_layer), classes)
batches = make_batches(trainX, 100, classes, trainY)
validation_batch = make_batch(textX, classes, testY)

params = HyperParams(epochs=10, learning_rate=0.4, momentum=0.7)
fit!(
    nn,
    params,
    batches,
    validation_batch,
    FitConfig(save_file="simple.jld")
)

predictions = predict(nn, textX, testY)
println("test error: ", test_error(predictions, testY))
