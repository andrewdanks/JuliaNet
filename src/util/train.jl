function get_linked_layers(layers::Vector{NeuralLayer})
    linked_layers = [LinkedLayer(layer) for layer in layers]

    linked_layers[1].next = linked_layers[2]
    for i = 2:length(linked_layers)-1
        linked_layers[i].prev = linked_layers[i-1]
        linked_layers[i].next = linked_layers[i+1]
    end
    linked_layers[end].prev = linked_layers[end-1]

    linked_layers
end


function get_linked_layers(
    layers::Vector{NeuralLayer},
    prev_linked_layers::Array{LinkedLayer}
)
    linked_layers = get_linked_layers(layers)
    for (i, prev_ll) in enumerate(prev_linked_layers)
        layer = linked_layers[i]
        layer.prev_weight_delta = prev_ll.weight_delta
    end
    linked_layers
end


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
