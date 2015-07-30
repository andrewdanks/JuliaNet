# Types applicable only to training

type FitHistory
    training_loss::Vector{T_FLOAT}
    training_classification_error::Vector{T_FLOAT}
    validation_loss::Vector{T_FLOAT}
    validation_classification_error::Vector{T_FLOAT}

    FitHistory() = new(T_FLOAT[], T_FLOAT[], T_FLOAT[], T_FLOAT[])
end


function Base.show(history::FitHistory)
    function format(val::T_FLOAT)
        round(val, 5)
    end
    println("==== EPOCH ", length(history.training_loss), " ====")

    println(
        "Training Loss:\t\t\t\t",
        format(history.training_loss[end])
    )
    if length(history.training_classification_error) > 0
        println(
            "Training Classification Error:\t\t",
            format(history.training_classification_error[end])
        )
    end
    if length(history.validation_loss) > 0
        println(
            "Validation Loss:\t\t\t",
            format(history.validation_loss[end])
        )
        if length(history.validation_classification_error) > 0
            println(
                "Validation Classification Error:\t",
                format(history.validation_classification_error[end])
            )
        end
    end
end


function record_training_history!(history::FitHistory, loss::T_FLOAT, classification_error::T_FLOAT)
    push!(history.training_loss, loss)
    push!(history.training_classification_error, classification_error)
end

function record_validation_history!(history::FitHistory, loss::T_FLOAT, classification_error::T_FLOAT)
    push!(history.validation_loss, loss)
    push!(history.validation_classification_error, classification_error)
end

function record_training_history!(history::FitHistory, loss::T_FLOAT)
    push!(history.training_loss, loss)
end

function record_validation_history!(history::FitHistory, loss::T_FLOAT)
    push!(history.validation_loss, loss)
end

immutable type BatchResults
    grad_weights
    loss
    classification_error
end


immutable type EpochResults
    loss
    classification_error
end


function +(a::BatchResults, b::BatchResults)
    if isdefined(a, :classification_error)
        BatchResults(
            a.grad_weights + b.grad_weights,
            a.loss + b.loss,
            a.classification_error + b.classification_error
        )
    else
        BatchResults(
            a.grad_weights + b.grad_weights,
            a.loss + b.loss
        )
    end
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


function get_batch_chunks(batches::Vector{Batch})
    batch_chunks = Vector[]
    for batch in batches
        batch_size = size(batch)
        half_batch_size = int(batch_size/2)
        chunk1 = get_chunk_from_batch(batch, UnitRange(1, half_batch_size))
        chunk2 = get_chunk_from_batch(batch, UnitRange(1+half_batch_size, batch_size))
        push!(batch_chunks, [chunk1, chunk2])
    end
    batch_chunks
end


immutable type EarlyStopCriterion
    current_epoch::T_UINT
    history::FitHistory
end


immutable type ParamUpdateCriterion
    params::HyperParams
    current_epoch::T_UINT
    history::FitHistory
end


immutable type FitResults
    training_loss_history::Array{T_FLOAT, 1}
    validation_loss_history::Array{T_FLOAT, 1}
    time::T_FLOAT
end