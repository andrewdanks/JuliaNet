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


function get_batches(
    classes::Vector,
    num_batches::T_INT,
    batch_size::T_INT,
    data::Matrix{T_FLOAT},
    target_classes::Vector,
    input_map_size
)
    batches = Batch[]
    for i = 1:num_batches
        batch = get_ith_batch(data, i, batch_size)
        batch_target_classes = get_ith_batch(target_classes, i, batch_size)
        batch_target_output = get_target_output_matrix(classes, batch_target_classes)
        push!(batches, Batch(
            InputTensor(batch, input_map_size),
            batch_target_output,
            batch_target_classes
        ))
    end
    batches
end