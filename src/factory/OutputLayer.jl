function FullyConnectedOutputLayer(
    input_layer::NeuralLayer,
    num_classes::T_INT,
    weight_sampler::Function,
    activator::Activator=SOFTMAX_ACTIVATOR
)
    HiddenLayer(
        input_layer,
        int(ones(input_layer.size, num_classes)),
        activator,
        weight_sampler
    )
end
