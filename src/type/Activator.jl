immutable type Activator
    activation_fn::Function
    grad_activation_fn::Function
end


IDENTITY_ACTIVATOR = Activator(identity, identity)
SIGMOID_ACTIVATOR = Activator(sigmoid, grad_sigmoid)
TANH_ACTIVATOR = Activator(tanh, grad_tanh)
LECUN_TANH_ACTIVATOR = Activator(lecun_tanh, grad_lecun_tanh)
SOFTMAX_ACTIVATOR = Activator(softmax, (x) -> 1)
