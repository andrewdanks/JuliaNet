module JuliaNet

import Base.push!
import Base.size

include("constants.jl")

include("util/activation.jl")

# Data types
include("type/HyperParams.jl")
include("type/Activator.jl")
include("type/InputTensor.jl")
include("type/NeuralLayer.jl")
include("type/HiddenLayer.jl")
include("type/NeuralNetwork.jl")
include("type/misc.jl")

# Factories for the data types
include("factory/HiddenLayer.jl")
include("factory/OutputLayer.jl")
include("factory/NeuralNetwork.jl")

include("util/train.jl")
include("util/misc.jl")
include("util/matrix.jl")

include("train.jl")
include("score.jl")
include("regularization.jl")
include("early_stopping.jl")
include("loss.jl")
include("preprocess.jl")

export fit!,
       predict,
       get_output,
       test_error,
       sample_weights,
       accuracy

export NeuralNetwork,
       HiddenLayer,
       OutputLayer,
       ConvolutionalLayer,
       PoolingLayer,
       MaxPoolingLayer,
       MaxPoolingActivator,
       HyperParams

export SimpleFullyConnectedNeuralNetwork,
       FullyConnectedHiddenLayers,
       FullyConnectedOutputLayer,
       FullyConnectedHiddenAndOutputLayers

export cross_entropy_loss
       mean_squared_error

export zero_mean,
       unit_variance

export IDENTITY_ACTIVATOR,
       SIGMOID_ACTIVATOR,
       TANH_ACTIVATOR,
       SOFTMAX_ACTIVATOR

export rand_range

end