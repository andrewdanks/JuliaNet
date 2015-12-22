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


function default_weight_sampler(dims::Tuple{T_UINT, T_UINT})
    default_weight_sampler(dims[1], dims[2])
end


function default_weight_sampler(rows::T_UINT, cols::T_UINT=1)
    val = 4 * sqrt(6 / (rows + cols))
    rand_range(-val, val, rows, cols)
end


function make_weights(dims::T_2D, weight_sampler::Function=default_weight_sampler)
    weight_sampler(dims)
end


function make_weights(dims::T_4D, weight_sampler::Function=default_weight_sampler)
    weights = zeros(dims)
    for i = 1:dims[2]
        for j = 1:dims[1]
            weights[j, i, :, :] = weight_sampler(dims[3], dims[4])
        end
    end
    weights
end


function get_target_output_matrix(
    classes::Vector,
    target_classes::Vector
)
    num_classes = length(classes)
    output_size = length(target_classes)
    target_output = zeros(num_classes, output_size)
    for j = 1:output_size
        target = zeros(num_classes)
        target[findfirst(classes, target_classes[j])] = 1.
        target_output[:, j] = target
    end
    target_output
end


function make_batch(
    data::T_TENSOR,
    classes::Vector,
    target_classes::Vector,
    input_map_size=T_NONE
)
    if input_map_size != T_NONE
        data_tensor = InputTensor(data, input_map_size)
    else
        data_tensor = InputTensor(data)
    end
    make_batch(data_tensor, 1:length(target_classes), classes, target_classes)
end


function make_batch(
    data::InputTensor,
    range::UnitRange{T_INT},
    classes::Vector,
    target_classes::Vector
)
    Batch(
        InputTensor(input_range(data, range)),
        get_target_output_matrix(classes, target_classes[range]),
        target_classes[range]  
    ) 
end


function make_batches(
    data::T_TENSOR,
    batch_size::T_INT,
    classes::Vector,
    target_classes::Vector,
    input_map_size=T_NONE
)
    batches = Batch[]
    if input_map_size != T_NONE
        data_tensor = InputTensor(data, input_map_size)
    else
        data_tensor = InputTensor(data)
    end
    i = 1
    while i <= data_tensor.batch_size
        range = i:min(i + batch_size - 1, data_tensor.batch_size)
        push!(batches, make_batch(data_tensor, range, classes, target_classes))
        i += batch_size
    end
    batches
end


function make_batch(
    data::T_TENSOR,
    input_map_size=T_NONE
)
    if input_map_size != T_NONE
        input = InputTensor(data, input_map_size)
    else
        input = InputTensor(data)
    end
    Batch(input, vectorized_data(input))
end


function make_batches(
    data::T_TENSOR,
    batch_size::T_INT,
    input_map_size=T_NONE
)
    batches = Batch[]
    data_tensor = InputTensor(data)
    i = 1
    while i <= data_tensor.batch_size
        range = i:min(i + batch_size - 1, data_tensor.batch_size)
        input = input_range(data_tensor, range)
        push!(batches, make_batch(input, input_map_size))
        i += batch_size
    end
    batches
end