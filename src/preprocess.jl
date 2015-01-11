function zero_mean(M::Matrix{T_FLOAT})
    _, num_cols = size(M)
    means = repmat(mean(M, 2), 1, num_cols)
    M - means
end

function unit_variance(M::Matrix{T_FLOAT})
    _, num_cols = size(M)
    std_devs = sqrt(var(M, 2))
    std_devs[find(x -> x == 0, std_devs)] = 1.
    std_devs = repmat(std_devs, 1, num_cols)
    M ./ std_devs
end