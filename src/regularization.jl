function set_dropout_masks!(layers::Vector{LinkedLayer}, batch_size::T_INT)
    for layer in layers
        if isdefined(layer.data_layer, :dropout_coefficient)
            set_dropout_mask!(layer, layer.data_layer.dropout_coefficient, batch_size)
        end
    end
end


function set_dropout_mask!(layer::LinkedLayer, coefficient::T_FLOAT, batch_size::T_INT)
    mask_size = (size(layer.data_layer.weights)[2], batch_size)
    layer.dropout_mask = get_dropout_mask(
        mask_size,
        coefficient
    )
end


function get_dropout_mask(size, prob)
    M = rand(size)
    M[find(x -> x < prob, M)] = 0.
    M[find(x -> x != 0, M)] = 1.
    M
end
