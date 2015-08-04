# Types applicable only to training

immutable type HyperParams
    epochs::T_INT
    learning_rate::T_FLOAT
    momentum::T_FLOAT
    nesterov::Bool

    function HyperParams(;
        epochs::T_UINT=1,
        learning_rate::T_FLOAT=0.1,
        momentum::T_FLOAT=0.,
        nesterov::Bool=false
    )
        new(epochs, learning_rate, momentum, nesterov)
    end
end

function Base.show(io::IO, params::HyperParams)
    println(io, "Epochs: ", params.epochs)
    println(io, "Learning Rate: ", params.learning_rate)
    println(io, "Momentum: ", params.momentum)
    println(io, "Nesterov: ", params.nesterov)
end


immutable type FitConfig
    save_file
    verbose::Bool
    parallelize::Bool

    function FitConfig(;
        save_file=nothing,
        verbose::Bool=true,
        parallelize::Bool=false
    )
        new(save_file, verbose, parallelize)
    end
end


immutable type BatchResults
    grad_weights
    loss
    classification_error
    BatchResults(grad_weights, loss) = new(grad_weights, loss)
end


immutable type EpochResults
    loss
    classification_error
    EpochResults(loss) = new(loss)
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