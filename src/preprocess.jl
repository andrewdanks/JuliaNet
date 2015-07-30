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


function get_batch(
    data::T_TENSOR,
    classes::Vector,
    target_classes::Vector
)
    get_batch(InputTensor(data), 1:length(target_classes), classes, target_classes)
end


function get_batch(
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
    target_classes::Vector
)
    batches = Batch[]
    data_tensor = InputTensor(data)
    i = 1
    while i <= data_tensor.batch_size
        range = i:min(i + batch_size - 1, data_tensor.batch_size)
        push!(batches, get_batch(data_tensor, range, classes, target_classes))
        i += batch_size
    end
    batches
end


function make_batches(
    data::T_TENSOR,
    target_data::T_TENSOR,
    batch_size::T_INT
)
    batches = Batch[]
    data_tensor = InputTensor(data)
    i = 1
    data_size = size(data)[1]
    while i <= data_size
        range = i:(i+batch_size)
        push!(batches, Batch(
            InputTensor(input_range(data_tensor, range)),
            target_data[range, :], #todo: how to do this for higher dims?
        ))
        i += batch_size + 1
    end
    batches
end