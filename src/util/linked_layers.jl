function get_linked_layers{T<:NeuralLayer}(layers::Vector{T})
    linked_layers = [LinkedLayer{typeof(layer)}(layer) for layer in layers]

    if length(linked_layers) > 1
        linked_layers[1].next = linked_layers[2]
        for i = 2:length(linked_layers)-1
            linked_layers[i].prev = linked_layers[i-1]
            linked_layers[i].next = linked_layers[i+1]
        end
        linked_layers[end].prev = linked_layers[end-1]
    end

    linked_layers
end


function get_linked_layers(
    layers::Vector{NeuralLayer},
    prev_linked_layers::Array{LinkedLayer}
)
    linked_layers = get_linked_layers(layers)
    for (i, prev_ll) in enumerate(prev_linked_layers)
        layer = linked_layers[i]
        if isdefined(prev_ll, :weight_delta)
            layer.weight_delta = prev_ll.weight_delta
        end
    end

    linked_layers
end
