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
include("type/training.jl")
include("type/HiddenLayer.jl")
include("type/FeatureMapLayer.jl")
include("type/ConvolutionalLayer.jl")
include("type/PoolingLayer.jl")
include("type/NeuralNetwork.jl")

# Factories for the data types
include("factory/NeuralLayer.jl")

include("util/linked_layers.jl")
include("util/misc.jl")
include("util/matrix.jl")
include("util/convolution.jl")

include("regularization.jl")
include("train.jl")
include("score.jl")
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
       HyperParams,
       InputTensor

export FullyConnectedHiddenLayers,
       FullyConnectedOutputLayer,
       FullyConnectedHiddenAndOutputLayers

export cross_entropy_loss
       mean_squared_error

export zero_mean,
       unit_variance,
       get_batch,
       make_batches

export IDENTITY_ACTIVATOR,
       SIGMOID_ACTIVATOR,
       TANH_ACTIVATOR,
       SOFTMAX_ACTIVATOR

export rand_range

end
