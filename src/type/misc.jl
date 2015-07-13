# Types applicable only to training

type LinkedLayer
    activator::Activator
    weights::Matrix

    prev::LinkedLayer
    next::LinkedLayer

    input::InputTensor
    pre_activation::Matrix
    activation::InputTensor
    grad_weights::Matrix
    weight_delta::Matrix
    prev_weight_delta::Matrix

    LinkedLayer(activator, weights) = new(activator, weights)
end


function has_prev(layer::LinkedLayer)
    isdefined(layer, :prev)
end


function has_next(layer::LinkedLayer)
    isdefined(layer, :next)
end


type BatchResults
    grad_weights    
    loss
    classification_error
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