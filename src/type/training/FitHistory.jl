type FitHistory
    training_loss::Vector{T_FLOAT}
    training_classification_error::Vector{T_FLOAT}
    validation_loss::Vector{T_FLOAT}
    validation_classification_error::Vector{T_FLOAT}

    FitHistory() = new(T_FLOAT[], T_FLOAT[], T_FLOAT[], T_FLOAT[])
end


function Base.size(history::FitHistory)
    length(history.training_loss)
end


function show_epoch(io::IO, history::FitHistory, epoch_idx::T_UINT)
    function format(val::T_FLOAT)
        round(val, 5)
    end
    println(io, "==== EPOCH ", length(history.training_loss), " ====")

    println(
        io,
        "Training Loss:\t\t\t\t",
        format(history.training_loss[epoch_idx])
    )
    if length(history.training_classification_error) > 0
        println(
            io,
            "Training Classification Error:\t\t",
            format(history.training_classification_error[epoch_idx])
        )
    end
    if length(history.validation_loss) > 0
        println(
            io,
            "Validation Loss:\t\t\t",
            format(history.validation_loss[epoch_idx])
        )
        if length(history.validation_classification_error) > 0
            println(
                io,
                "Validation Classification Error:\t",
                format(history.validation_classification_error[epoch_idx])
            )
        end
    end
end


function Base.show(io::IO, history::FitHistory)
    for epoch = 1:size(history)
        show_epoch(io, history, epoch)
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
