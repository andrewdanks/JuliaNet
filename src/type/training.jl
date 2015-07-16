# Types applicable only to training

type LinkedLayer
    data_layer::NeuralLayer

    prev::LinkedLayer
    next::LinkedLayer

    input::InputTensor
    pre_activation::T_TENSOR
    activation::InputTensor
    grad_weights::T_TENSOR
    weight_delta::T_TENSOR

    prev_weight_delta::T_TENSOR
    dropout_mask::Matrix

    LinkedLayer(layer::NeuralLayer) = new(layer)
end


function has_prev(layer::LinkedLayer)
    isdefined(layer, :prev)
end


function has_next(layer::LinkedLayer)
    isdefined(layer, :next)
end


type BatchResults
    grad_weights::T_TENSOR
    loss::T_FLOAT
    classification_error::T_FLOAT
end


function +(a::BatchResults, b::BatchResults)
    BatchResults(
        a.grad_weights + b.grad_weights,
        a.loss + b.loss,
        a.classification_error + b.classification_error
    )
end


type Batch
    input::InputTensor
    target_output::Matrix
    target_classes::Vector
end


function Base.size(batch::Batch)
    batch.input.batch_size
end


type EarlyStopCriterion
    current_epoch::T_UINT
    training_loss_history::Vector{T_FLOAT}
    validation_loss_history::Vector{T_FLOAT}
end


type ParamUpdateCriterion
    params::HyperParams
    current_epoch::T_UINT
    training_loss_history::Vector{T_FLOAT}
    validation_loss_history::Vector{T_FLOAT}
end


type FitResults
    training_loss_history::Array{T_FLOAT, 1}
    validation_loss_history::Array{T_FLOAT, 1}
    time::T_FLOAT
end