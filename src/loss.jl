function mean_squared_error(
    actual_output::T_TENSOR,
    target_output::T_TENSOR
)
    0.5 * mean(sum((actual_output - target_output).^2, 1))
end


function cross_entropy_error(
    actual_output::T_TENSOR,
    target_output::T_TENSOR
)
    num_cases = size(target_output)[2]
    -sum(sum(target_output .* log(actual_output), 1)) / num_cases
end
