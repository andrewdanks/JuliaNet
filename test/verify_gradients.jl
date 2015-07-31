using JuliaNet
using Base.Test

EPSILON = 1e-5
ERROR_THRESHOLD = 1e-7
LOSS_FN = mean_squared_error

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


function nn_output_finite_difference(nn, layer_idx, epsilon, batch, idx)
    nn.layers[layer_idx].weights[idx] += epsilon
    JuliaNet.forward_pass(nn, batch.input)
end


function verify_gradients(nn, linked_layers, params, batch)
    num_bad_grads = 0
    total = 0
    sum_bad_e = 0.

    for L = 1:length(nn.layers)
        layer = linked_layers[L]
        num_weights = length(layer.data_layer.weights)
        for i = 1:num_weights
            output1 = nn_output_finite_difference(nn, L, EPSILON, batch, i)
            output2 = nn_output_finite_difference(nn, L, -EPSILON, batch, i)

            loss1 = LOSS_FN(output1, batch.target_output)
            loss2 = LOSS_FN(output2, batch.target_output)

            delta = (loss1 - loss2) / (2 * EPSILON)

            e = abs(delta - layer.grad_weights[i])

            relative_error = e / (abs(delta) + abs(layer.grad_weights[i]))

            if e > ERROR_THRESHOLD
                #println("L", L, "; (",i,"); ", e)
                num_bad_grads += 1
                sum_bad_e += e
            end
            total += 1
        end
    end

    if num_bad_grads > 0
        println("num_bad_grads: ", num_bad_grads)
        println("mean_bad_e: ", sum_bad_e / num_bad_grads)
        println("sum_bad_e: ", sum_bad_e)
        println("total: ", total)
        false
    else
        println("Success!")
        true
    end
end


function verify_networks(nns::Vector{NeuralNetwork})
    params = HyperParams()
    params.learning_rate = 1.0
    params.epochs = 1
    for nn in nns
        my_nn = deepcopy(nn)
        linked_layers = JuliaNet.get_linked_layers(nn.layers)
        batch = make_batch(X, classes, Y)
        JuliaNet.fit_batch!(linked_layers, batch)
        is_correct = verify_gradients(my_nn, linked_layers, params, batch)
        println("=================================")
    end
end


function make_simple_network()
    hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
        num_features,
        [15, 25],
        num_classes,
        SIGMOID_ACTIVATOR,
        SOFTMAX_ACTIVATOR,
        JuliaNet.default_weight_sampler
    )
    NeuralNetwork(vcat(hidden_layers, output_layer))
end


verify_networks([
    make_simple_network(),
])

