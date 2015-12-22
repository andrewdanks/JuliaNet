function rand_range(min::T_FLOAT, max::T_FLOAT, rows::T_INT, cols::T_INT)
    (max - min) * rand(rows, cols) + min
end


function sample_weights(rows::T_INT, cols::T_INT=1)
    val = 4 * sqrt(6 / (rows + cols))
    rand_range(-val, val, rows, cols)
end


function sample_weights(dims::T_2D)
    sample_weights(dims[1], dims[2])
end


function to_int(a::AbstractArray)
	round(T_INT, a)
end
