using JuliaNet
using Base.Test

EPSILON = 1e-6
ERROR_THRESHOLD = 1e-5

srand(12345)
input_dimensions = (8, 8)
NUM_FEATURES = prod(input_dimensions)
X = rand(NUM_FEATURES, 10);
Y = int(randbool(10))

# using MNIST
# NUM_FEATURES = 28 * 28
# MNIST_X, MNIST_Y = MNIST.testdata()
# X, Y= MNIST_X[:, 1:100], MNIST_Y[1:100]

classes = sort(unique(Y))
NUM_CLASSES = length(classes)


function get_finite_diff_output(nn, layer_idx, epsilon, batch, idx)
    nn.layers[layer_idx].weights[idx] += epsilon
    JuliaNet.forward_pass(nn, batch.input)
end


function verify_gradients(nn, linked_layers, batch, loss_fn)
    num_bad_grads = 0
    total = 0
    sum_bad_e = 0.

    for L = 1:length(nn.layers)
        layer = linked_layers[L]
        num_weights = length(layer.data_layer.weights)
        for i = 1:num_weights
            output1 = get_finite_diff_output(nn, L, EPSILON, batch, i)
            output2 = get_finite_diff_output(nn, L, -2*EPSILON, batch, i)

            loss1 = loss_fn(output1, batch.target_output)
            loss2 = loss_fn(output2, batch.target_output)

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
        println("GRADIENT CHECK FAILED:")
        println("num_bad_grads: ", num_bad_grads)
        println("mean_bad_e: ", sum_bad_e / num_bad_grads)
        println("sum_bad_e: ", sum_bad_e)
        println("total: ", total)
        false
    else
        true
    end
end


function verify_network(nn, batch, loss_fn)
    linked_layers = JuliaNet.get_linked_layers(nn.layers)
    @time JuliaNet.fit_batch!(linked_layers, batch)
    is_correct = verify_gradients(nn, linked_layers, batch, loss_fn)
end


function verify_simple_network()
    hidden_layers, output_layer = FullyConnectedHiddenAndOutputLayers(
        NUM_FEATURES,
        [15, 25],
        NUM_CLASSES,
        :sigmoid,
        :softmax,
        JuliaNet.default_weight_sampler
    )
    nn = NeuralNetwork(vcat(hidden_layers, output_layer))
    batch = make_batch(X, classes, Y)
    verify_network(nn, batch, cross_entropy_error)
end


function verify_autoencoder()
    ae = Autoencoder(NUM_FEATURES, 25, :sigmoid)
    batch = make_batch(X)
    verify_network(ae, batch, mean_squared_error)
end


verify_simple_network()
verify_autoencoder()