using JuliaNet
using Base.Test

EPSILON = 1e-6
ERROR_THRESHOLD = 1e-7
LOSS_FN = cross_entropy_loss
#LOSS_FN = JuliaNet.mean_squared_error

srand(12345)
input_dimensions = (16, 16)
num_features = prod(input_dimensions)
X = rand(num_features, 20);
Y = int(randbool(20))

# using MNIST
# num_features = 28 * 28
# MNIST_X, MNIST_Y = MNIST.testdata()
# X, Y= MNIST_X[:, 1:100], MNIST_Y[1:100]

classes = sort(unique(Y))
num_classes = length(classes)

function nn_output_finite_difference(nn::NeuralNetwork, params, layer_idx, epsilon, input_tensor, idx)
    nn_copy = deepcopy(nn)
    layer = JuliaNet.layer_at_index(nn_copy, layer_idx)
    layer.weights[idx] += epsilon
    JuliaNet.forward_pass!(nn_copy, params, input_tensor)
end

function verify_gradients(nn::NeuralNetwork, params, input_tensor, y, target_output)
    num_bad_grads = 0
    total = 0
    sum_bad_e = 0.

    ignored_layers = [PoolingLayer]

    for L = 2:JuliaNet.num_layers(nn)
        layer = deepcopy(JuliaNet.layer_at_index(nn, L))
        if !(typeof(layer) in ignored_layers)
            num_weights = length(layer.weights)
            for i = 1:num_weights
                output1 = nn_output_finite_difference(nn, params, L, EPSILON, input_tensor, i)
                output2 = nn_output_finite_difference(nn, params, L, -EPSILON, input_tensor, i)

                loss1 = LOSS_FN(output1, target_output)
                loss2 = LOSS_FN(output2, target_output)

                delta = (loss1 - loss2) / (2 * EPSILON)

                e = abs(delta - layer.grad_weights[i])

                if e > ERROR_THRESHOLD
                    #println("L", L, "; (",i,"); ", e)
                    num_bad_grads += 1
                    sum_bad_e += e
                end
                total += 1
            end
        end
    end

    if num_bad_grads > 0
        println("num_bad_grads: ", num_bad_grads)
        println("mean_bad_e: ", sum_bad_e / num_bad_grads)
        println("sum_bad_e: ", sum_bad_e)
        println("total: ", total)
    else
        println("Success!")
    end
end

function verify_networks(nns::Vector{NeuralNetwork})

    params = HyperParams()
    params.learning_rate = 1.0
    params.epochs = 1

    for batch_size in [100]
        params.batch_size = batch_size
        for momentum in [0.]
            params.momentum = momentum
            for nn in nns
                my_nn = deepcopy(nn)
                input_tensor = JuliaNet.format_input(JuliaNet.input_layer(my_nn), X)
                output = JuliaNet.forward_pass!(my_nn, params, input_tensor)
                target_output = JuliaNet.target_output_matrix(classes, Y)
                JuliaNet.backward_pass!(my_nn, params, output, target_output)
                verify_gradients(my_nn, params, input_tensor, Y, target_output)
                println("=================================")
            end
        end
    end
end

function make_simple_network()
    srand(123)
    input_layer = InputLayer(num_features)
    hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
        input_layer,
        [15, 25],
        num_classes,
        sample_weights,
        SIGMOID_ACTIVATOR
    )
    NeuralNetwork(vcat(input_layer, hidden_layers, output_layer))
end

function make_simple_cnn()
    srand(123)
    input_layer = InputLayer(input_dimensions)

    conv_layer1 = ConvolutionalLayer(
        input_layer,
        (5, 5), 4,
        TANH_ACTIVATOR,
        sample_weights
    )

    pooling_layer1 = PoolingLayer(conv_layer1, (2, 2))

    cnn_layers = vcat(conv_layer1, pooling_layer1)
    output_layer = FullyConnectedOutputLayer(cnn_layers[end], num_classes, sample_weights)

    NeuralNetwork(vcat(input_layer, cnn_layers, output_layer))
end

function make_double_cnn()
    srand(123)
    input_layer = InputLayer(input_dimensions)

    conv_layer1 = ConvolutionalLayer(
        input_layer,
        (5, 5), 4,
        TANH_ACTIVATOR,
        sample_weights
    )

    pooling_layer1 = PoolingLayer(conv_layer1, (2, 2))

    conv_layer2 = ConvolutionalLayer(
        pooling_layer1,
        (5, 5), 6,
        TANH_ACTIVATOR,
        sample_weights
    )

    pooling_layer2 = PoolingLayer(conv_layer2, (2, 2))

    cnn_layers = vcat(conv_layer1, pooling_layer1, conv_layer2, pooling_layer2)
    output_layer = FullyConnectedOutputLayer(cnn_layers[end], num_classes, sample_weights)

    NeuralNetwork(vcat(input_layer, cnn_layers, output_layer))
end

function make_lenet()
    srand(123)
    input_dimensions = (28, 28)

    input_layer = InputLayer(input_dimensions)

    conv_layer1 = ConvolutionalLayer(
        input_layer.feature_map_size,
        input_layer.num_maps,
        (5, 5), 4,
        TANH_ACTIVATOR,
        sample_weights
    )

    pooling_layer1 = PoolingLayer(
        conv_layer1.feature_map_size,
        conv_layer1.num_maps,
        (2, 2)
    )

    conv_layer2 = ConvolutionalLayer(
        pooling_layer1.feature_map_size,
        conv_layer1.num_maps,
        (5, 5), 6,
        TANH_ACTIVATOR,
        sample_weights
    )

    pooling_layer2 = PoolingLayer(
        conv_layer2.feature_map_size,
        conv_layer2.num_maps,
        (2, 2)
    )

    cnn_layers = vcat(conv_layer1, pooling_layer1, conv_layer2, pooling_layer2)

    hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
        cnn_layers[end],
        [5, 10],
        num_classes,
        sample_weights
    )

    NeuralNetwork(input_layer, vcat(cnn_layers, hidden_layers), output_layer)
end

verify_networks([
    make_simple_network(),
    #make_simple_cnn(),
    make_double_cnn(),
    #make_lenet()
])

