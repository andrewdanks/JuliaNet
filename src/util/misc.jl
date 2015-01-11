function rand_range(min::T_FLOAT, max::T_FLOAT, rows::T_INT, cols::T_INT)
    (max - min) * rand(rows, cols) + min
end

function sample_weights(rows::T_INT, cols::T_INT=1)
    val = 4 * sqrt(6 / (rows + cols))
    rand_range(-val, val, rows, cols)
end

function sample_weights(dims::(T_INT, T_INT))
    sample_weights(dims[1], dims[2])
end