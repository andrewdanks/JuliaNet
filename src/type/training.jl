# Types applicable only to training

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
        InputTensor(get_batch_range(batch.input, range)),
        batch.target_output[:, range],
        batch.target_classes[range]
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