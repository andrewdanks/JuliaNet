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
    layer = nn_copy.layers[layer_idx]
    layer.weights[idx] += epsilon
    JuliaNet.forward_pass!(nn_copy, params, input_tensor)
end


function verify_gradients(nn::NeuralNetwork, params, input_tensor, y, target_output)
    num_bad_grads = 0
    total = 0
    sum_bad_e = 0.

    ignored_layers = [PoolingLayer]

    for L = 2:size(nn)
        layer = deepcopy(nn.layers[L])
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
    hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
        num_features,
        [15, 25],
        num_classes,
        sample_weights,
        SIGMOID_ACTIVATOR
    )
    NeuralNetwork(vcat(hidden_layers, output_layer))
end


verify_networks([
    make_simple_network(),
])

