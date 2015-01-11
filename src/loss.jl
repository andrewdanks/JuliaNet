function cross_entropy_loss(
    actual_output::Matrix{T_FLOAT},
    target_output::Matrix{T_FLOAT}
)
    _, num_cases = size(target_output)
    -sum(sum(target_output .* log(actual_output))) / num_cases
end

function mean_squared_error(
    actual_output::Matrix{T_FLOAT},
    target_output::Matrix{T_FLOAT}
)
    0.5 .* mean(sum((actual_output - target_output).^2, 1))
end