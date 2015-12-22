module JuliaNet

import Base.+
import Base.==
import Base.push!
import Base.size

include("constants.jl")

include("util/activation.jl")

include("type/InputTensor.jl")
include("preprocess.jl")
include("type/training/Batch.jl")
include("type/training/FitHistory.jl")
include("type/training/other.jl")
include("type/layers/NeuralLayer.jl")
include("type/layers/HiddenLayer.jl")
include("type/layers/FeatureMapLayer.jl")
include("type/layers/ConvolutionalLayer.jl")
include("type/layers/PoolingLayer.jl")
include("type/NeuralNetwork.jl")

# Factories for the data types
include("factory/NeuralLayer.jl")

include("util/linked_layers.jl")
include("util/misc.jl")
include("util/matrix.jl")
include("util/convolution.jl")

include("train.jl")
include("score.jl")
include("early_stopping.jl")
include("loss.jl")

export fit!,
       forward_pass,
       predict,
       get_output,
       test_error,
       sample_weights,
       accuracy

export NeuralNetwork,
       Autoencoder,
       StackedAutoencoder,
       HiddenLayer,
       OutputLayer,
       ConvolutionalLayer,
       PoolingLayer,
       HyperParams,
       FitConfig,
       InputTensor

export FullyConnectedHiddenLayers,
       FullyConnectedOutputLayer,
       FullyConnectedHiddenAndOutputLayers

export cross_entropy_error,
       mean_squared_error

export zero_mean,
       unit_variance,
       make_batch,
       make_batches

export sigmoid, grad_sigmoid, grad_tanh, grad_squared_error

export rand_range

end
