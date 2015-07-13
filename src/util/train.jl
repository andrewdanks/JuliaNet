function get_linked_layers(layers::Vector{NeuralLayer})
    linked_layers = [
        LinkedLayer(layer.activator, layer.weights)
        for layer in layers
    ]

    linked_layers[1].next = linked_layers[2]
    for i = 2:length(linked_layers)-1
        linked_layers[i].prev = linked_layers[i-1]
        linked_layers[i].next = linked_layers[i+1]
    end
    linked_layers[end].prev = linked_layers[end-1]

    linked_layers
end

function get_linked_layers(layers::Vector{NeuralLayer}, prev_linked_layers::Vector{LinkedLayer})
    linked_layers = get_linked_layers(layers)
    for (i, prev_ll) in enumerate(prev_linked_layers)
        linked_layers[i].prev_weight_delta = prev_ll.weight_delta
    end
    linked_layers
end