function FullyConnectedOutputLayer(
    input_layer::NeuralLayer,
    num_classes::T_INT,
    weight_sampler::Function,
    activator::Activator=SOFTMAX_ACTIVATOR
)
    HiddenLayer(
        int(ones(size(input_layer), num_classes)),
        activator,
        weight_sampler
    )
end
