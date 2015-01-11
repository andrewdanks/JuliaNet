type Patch
    row_range::UnitRange{T_INT}
    col_range::UnitRange{T_INT}
end

function set_patch_weights!(
    weights::Matrix{T_FLOAT},
    input_dimensions::(T_INT, T_INT),
    patch_idx::T_INT,
    patch::Patch,
    patch_weights::Matrix{T_FLOAT}
)
    patch_weights0 = zeros(input_dimensions)
    patch_weights0[patch.row_range,patch.col_range] = patch_weights
    weights[:,patch_idx] = vec(patch_weights0)
end

function patch_weight_matrix_dimensions(
    input_dimensions::(T_INT, T_INT),
    num_patches::T_INT
)
    (input_dimensions[1] * input_dimensions[2], num_patches)
end

function num_patches_overlapping(
    input_dimensions::(T_INT, T_INT),
    patch_dimensions::(T_INT, T_INT)
)
    (
        input_dimensions[1] - patch_dimensions[1] + 1,
        input_dimensions[2] - patch_dimensions[2] + 1,
    )
end

function overlapping_patches(
    num_patches::(T_INT, T_INT),
    input_dimensions::(T_INT, T_INT),
    patch_dimensions::(T_INT, T_INT)
)
    patch_range_fn = function(patch_idx::T_INT, patch_size::T_INT, input_dimension::T_INT)
        start = patch_idx

        stop_guess = patch_idx + patch_size - 1
        stop = min(stop_guess, input_dimension)
        
        subtract_from_start = max(stop_guess - input_dimension, 0)
        start -= subtract_from_start

        start:stop
    end

    make_patches(num_patches, patch_range_fn, input_dimensions, patch_dimensions)
end

function non_overlapping_patches(
    num_patches::(T_INT, T_INT),
    input_dimensions::(T_INT, T_INT),
    patch_dimensions::(T_INT, T_INT)
)
    patch_range_fn = function(patch_idx::T_INT, patch_size::T_INT, input_dimension::T_INT)
        start = (patch_idx - 1) * patch_size + 1
        stop = patch_idx * patch_size

        start:stop
    end

    make_patches(num_patches, patch_range_fn, input_dimensions, patch_dimensions)
end

function make_patches(
    num_patches::(T_INT, T_INT),
    patch_range_fn::Function,
    input_dimensions::(T_INT, T_INT),
    patch_dimensions::(T_INT, T_INT)
)
    patches = Patch[]
    for i = 1:num_patches[2]
        patch_col_range = patch_range_fn(i, patch_dimensions[2], input_dimensions[2])
        for j = 1:num_patches[1]
            patch_row_range = patch_range_fn(j, patch_dimensions[1], input_dimensions[1])
            push!(patches, Patch(patch_row_range, patch_col_range))
        end
    end
    patches
end