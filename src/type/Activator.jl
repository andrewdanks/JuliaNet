const symbol_to_activator = {
    :identity => identity,
    :sigmoid => sigmoid,
    :tanh => tanh,
    :lecun_tanh => lecun_tanh,
    :softmax => softmax
}


const symbol_to_∇activator = {
    :identity => identity,
    :sigmoid => grad_sigmoid,
    :tanh => grad_tanh,
    :lecun_tanh => grad_lecun_tanh,
    :softmax => (x) -> 1
}


function get_activator(symbol::Symbol)
    symbol_to_activator[symbol]
end


function get_∇activator(symbol::Symbol)
    symbol_to_∇activator[symbol]
end
