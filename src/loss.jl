function mean_squared_error(
    actual_output::T_TENSOR,
    target_output::T_TENSOR
)
    0.5 * mean(sum((actual_output - target_output).^2, 1))
end
