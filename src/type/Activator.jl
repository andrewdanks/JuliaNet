immutable type Activator
    # Activation functions to be used at train time
    activation_fn::Function
    grad_activation_fn::Function

    # Activation function to be used at test time
    test_activation_fn::Function

    function Activator(activation_fn::Function, grad_activation_fn::Function)
        new(activation_fn, grad_activation_fn, activation_fn)
    end

    function Activator(activation_fn::Function, grad_activation_fn::Function, test_activation_fn::Function)
        new(activation_fn, grad_activation_fn, test_activation_fn)
    end
end

IDENTITY_ACTIVATOR = Activator(identity, identity)
SIGMOID_ACTIVATOR = Activator(sigmoid, grad_sigmoid)
TANH_ACTIVATOR = Activator(tanh, grad_tanh)
LECUN_TANH_ACTIVATOR = Activator(lecun_tanh, grad_lecun_tanh)
RELU_ACTIVATOR = Activator(rectified_linear, grad_rectified_linear)
SOFTMAX_ACTIVATOR = Activator(softmax, grad_squared_error)

function DropoutActivator(
    activation_fn::Function,
    grad_activation_fn::Function,
    dropout_probability::T_FLOAT
)
    dropout_activation_fn = function(input)
        zero_out_with_prob(activation_fn(input), dropout_probability)
    end
    test_dropout_activation_fn = function(input)
        activation_fn(input) .* (1 - dropout_probability)
    end
    Activator(dropout_activation_fn, grad_activation_fn, test_dropout_activation_fn)
end

function DropoutActivator(activator::Activator, dropout_probability::T_FLOAT)
    DropoutActivator(activator.activation_fn, activator.grad_activation_fn, dropout_probability)
end

function DropoutActivator(dropout_probability::T_FLOAT)
    DropoutActivator(IDENTITY_ACTIVATOR, dropout_probability)
end

function Base.show(io::IO, activator::Activator)
    println(io, typeof(activator), "[", activation_fn, "]")
end