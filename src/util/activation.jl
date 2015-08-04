function sigmoid(x)
    1 ./ (1. + exp(-x))
end


function grad_sigmoid(x)
    sig = sigmoid(x)
    sig .* (1. - sig)
end


function grad_tanh(x)
    1. - tanh(x).^2
end


function lecun_tanh(x)
    1.7159 * tanh(2/3 * x)
end


function grad_lecun_tanh(x)
    #1.14393 * (1 - tanh(2/3 * x))  
    1.7159 * 2/3 * (1 - 1/(1.7159)^2 * x .^ 2)
end


function softmax(input::Matrix{T_FLOAT})
    normalizer = log_sum_exp_over_rows(input)
    log_prob = input .- normalizer
    exp(log_prob)
end


function log_sum_exp_over_rows(A::Matrix{T_FLOAT})
    maxs_small = maximum(A, 1)
    maxs_big = repmat(maxs_small, size(A)[1], 1)
    log(sum(exp(A - maxs_big), 1)) + maxs_small
end


function grad_squared_error(
    actual_output::Matrix{T_FLOAT},
    target_output::Matrix{T_FLOAT}
)
    actual_output - target_output
end


const symbol_to_activator = {
    :identity => identity,
    :sigmoid => sigmoid,
    :tanh => tanh,
    :lecun_tanh => lecun_tanh,
    :softmax => softmax
}


const symbol_to_∇activator = {
    :identity => identity,
    :sigmoid => grad_sigmoid,
    :tanh => grad_tanh,
    :lecun_tanh => grad_lecun_tanh,
    :softmax => (x) -> 1
}


function get_activator(symbol::Symbol)
    symbol_to_activator[symbol]
end


function get_∇activator(symbol::Symbol)
    symbol_to_∇activator[symbol]
end
