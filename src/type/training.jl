# Types applicable only to training

type LinkedLayer{T<:NeuralLayer}
    data_layer::T

    prev::LinkedLayer
    next::LinkedLayer

    input::InputTensor
    pre_activation::T_TENSOR
    activation::InputTensor
    grad_weights::T_TENSOR
    weight_delta::T_TENSOR

    prev_weight_delta::T_TENSOR
    dropout_mask::Matrix

    # for pooling layer only
    max_masks::T_TENSOR

    LinkedLayer(layer::T) = new(layer)
end


function has_prev(layer::LinkedLayer)
    isdefined(layer, :prev)
end


function has_next(layer::LinkedLayer)
    isdefined(layer, :next)
end


immutable type BatchResults
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


immutable type Batch
    input::InputTensor
    target_output::Matrix
    target_classes::Vector
end


function Base.size(batch::Batch)
    batch.input.batch_size
end


function get_chunk_from_batch(batch::Batch, range::UnitRange{T_INT})
    Batch(
        InputTensor(get_batch_range(batch.input, range), batch.input.num_maps),
        target_output[range,:],
        target_classes[range]
    )
end


immutable type EarlyStopCriterion
    current_epoch::T_UINT
    training_loss_history::Vector{T_FLOAT}
    validation_loss_history::Vector{T_FLOAT}
end


immutable type ParamUpdateCriterion
    params::HyperParams
    current_epoch::T_UINT
    training_loss_history::Vector{T_FLOAT}
    validation_loss_history::Vector{T_FLOAT}
end


immutable type FitResults
    training_loss_history::Array{T_FLOAT, 1}
    validation_loss_history::Array{T_FLOAT, 1}
    time::T_FLOAT
end