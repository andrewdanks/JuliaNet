function no_stop_criterion_fn(
    criterion::EarlyStopCriterion
)
    false
end

function generalization_error(
    validation_loss::T_FLOAT,
    min_validation_loss::T_FLOAT
)
    100 * (validation_loss / min_validation_loss - 1)
end
