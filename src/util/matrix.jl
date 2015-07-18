function zero_out_with_prob(M::Matrix{T_FLOAT}, prob::T_FLOAT)
    M2 = rand(size(M))
    M2[find(x -> x < prob, M2)] = 0.
    M2[find(x -> x != 0, M2)] = 1.
    ret = M2 .* M
    ret
end


function get_ith_batch(X::Matrix, i::T_INT, batch_size::T_INT)
    start = (i - 1) * batch_size + 1
    finish = start + batch_size - 1
    X[:, start:finish]
end

function get_ith_batch(v::Vector, i::T_INT, batch_size::T_INT)
    vectorize(get_ith_batch(matrixify(v)', i, batch_size))
end

function num_data_matrix_inputs(T::T_2D_TENSOR)
    size(T)[2]
end

function num_data_matrix_features(T::T_2D_TENSOR)
    size(T)[1]
end

function squared_length(v::Vector)
    sum(v .^ 2)
end

function zeroify(T::T_TENSOR)
    zeros(size(T))
end

function matrixify(v::Vector)
    repmat(v, 1, 1)
end

function matrixify(x::Number)
    repmat([x], 1, 1)
end

function vectorize(M::Matrix)
    M[:]
end

function flipall(T::T_TENSOR)
    new_T = copy(T)
    for i = 1:ndims(T)
        new_T = flipdim(new_T, i)
    end
    new_T
end

function squeezeall(T::T_TENSOR)
    size_T = size(T)
    new_size_T = T_INT[]
    for i = 1:length(size_T)
        push!(new_size_T, size_T[i])
    end
    idxs = find(x -> x == 1, new_size_T)
    squeeze(T, idxs)
end

