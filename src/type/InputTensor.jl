type InputTensor
    # This data type provides abstraction to data going
    # into layers, which can be in a variety of dimensions

    data::T_4D_TENSOR
    batch_size::T_INT
    num_features::T_INT
    num_rows::T_INT
    num_cols::T_INT
    num_maps::T_INT

    function InputTensor(input::T_4D_TENSOR)
        size_input = size(input)
        new(
            input,
            size_input[1],
            size_input[3] * size_input[4],
            size_input[3],
            size_input[4],
            size_input[2]
        )
    end

    function InputTensor(input::T_1D_TENSOR)
        InputTensor(matrixify(input))
    end

    function InputTensor(input::T_2D_TENSOR)
        num_features = num_data_matrix_features(input)
        reshape_dims = (1, num_features)
        InputTensor(input, reshape_dims)
    end

    function InputTensor(input::T_2D_TENSOR, reshape_dims::UnionType)
        InputTensor(input)
    end

    function InputTensor(input::T_2D_TENSOR, reshape_dims::(T_UINT, T_UINT))
        size_input = size(input)
        batch_size = size_input[2]
        new_input = zeros(batch_size, 1, size_input[1])
        for x = 1:batch_size
            new_input[x, 1, :] = vec(input[:, x])
        end
        InputTensor(new_input, reshape_dims)
    end

    function InputTensor(input::T_3D_TENSOR)
        size_input = size(input)
        InputTensor(input, (size_input[3], 1))
    end

    function InputTensor(input::T_3D_TENSOR, reshape_dims::(T_INT, T_INT))
        size_input = size(input)
        batch_size = size_input[1]
        num_maps = size_input[2]
        num_features = prod(reshape_dims)
        new_input = zeros(batch_size, num_maps, reshape_dims[1], reshape_dims[2])
        for x = 1:batch_size
            for n = 1:num_maps
                new_input[x, n, :, :] = reshape(input[x, n, :], reshape_dims)
            end
        end
        InputTensor(new_input)
    end

    function InputTensor(input_tensor::InputTensor)
        input_tensor
    end
end

function vectorized_data(input_tensor::InputTensor)
    new_data = zeros(
        input_tensor.num_features * input_tensor.num_maps,
        input_tensor.batch_size
    )
    for x = 1:input_tensor.batch_size
        new_data[:, x] = vec(input_tensor.data[x, :, :, :])
    end
    new_data
end

function input_data(input_tensor::InputTensor, input_idx::T_INT)
    @assert 1 <= input_idx <= input_tensor.batch_size
    @inbounds ret = squeeze(input_tensor.data[input_idx, :, :, :], 1)
    ret
end

function map_data(input_tensor::InputTensor, map_idx::T_INT)
    @assert 1 <= map_idx <= input_tensor.num_maps
    @inbounds ret = squeeze(input_tensor.data[:, map_idx, :, :], 2)
    ret
end

function input_map_data(input_tensor::InputTensor, input_idx::T_INT, map_idx::T_INT)
    @assert 1 <= input_idx <= input_tensor.batch_size
    @assert 1 <= map_idx <= input_tensor.num_maps
    @inbounds ret = squeeze(input_tensor.data[input_idx, map_idx, :, :], (1, 2))
    ret
end

function zero_out_with_prob(input::InputTensor, prob::T_FLOAT)
    if prob > 0
        new_input_data = zeros(size(input.data))
        for x = 1:input.batch_size
            for m = 1:input.num_maps
                map_input = squeeze(input.data[x, m, :, :], (1, 2))
                zero_out_map_input = zero_out_with_prob(map_input, prob)
                new_input_data[x, m, :, :] = zero_out_map_input
            end
        end
        InputTensor(new_input_data)
    else
        input
    end
end

function Base.size(input_tensor::InputTensor, i)
    size(input_tensor.data, i)
end

function Base.slicedim(input_tensor::InputTensor, d, i)
    slicedim(input_tensor.data, d, i)
end

function Base.getindex(input_tensor::InputTensor, args...)
    getindex(input_tensor.data, args...)
end

function ==(a::InputTensor, b::InputTensor)
    a.data == b.data
end

