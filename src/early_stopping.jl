function no_stop_criterion_fn(
    current_epoch::T_INT,
    training_loss::T_FLOAT,
    validation_loss::T_FLOAT,
    training_lost_history::Array{T_FLOAT},
    validation_lost_history::Array{T_FLOAT}
)
    false
end

function generalization_error(
    validation_loss::T_FLOAT,
    min_validation_loss::T_FLOAT
)
    100 * (validation_loss / min_validation_loss - 1)
end
