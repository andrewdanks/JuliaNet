abstract NeuralLayer


type LinkedLayer{T<:NeuralLayer}
    data_layer::T

    prev::LinkedLayer
    next::LinkedLayer

    input::InputTensor
    pre_activation::T_TENSOR
    activation::InputTensor
    grad_weights::T_TENSOR
    weight_delta::T_TENSOR

    prev_weight_delta::T_TENSOR

    # for pooling layer only
    max_masks::T_TENSOR

    # contrain this layers weights to be the same as this one
    tied_weights::LinkedLayer

    LinkedLayer(layer::T) = new(layer)
end


function has_prev(layer::LinkedLayer)
    isdefined(layer, :prev)
end


function has_next(layer::LinkedLayer)
    isdefined(layer, :next)
end


function update_weights!{T<:NeuralLayer}(layer::LinkedLayer{T}, params::HyperParams)
    if isdefined(layer, :weight_delta)
        prevΔ = layer.weight_delta
    else
        prevΔ = 0.0
    end

    η = params.learning_rate
    μ = params.momentum
    λ = params.L2_decay
    dx = layer.grad_weights
    n = layer.activation.batch_size

    if params.nesterov
        velocity = μ * prevΔ - η * dx
        Δ = -μ * prevΔ + (1 + μ) * velocity
    elseif μ > 0
        Δ = μ * prevΔ - η * dx
    else
        Δ = -η * dx
    end

    if λ > 0
        layer.data_layer.weights *= 1 - η*λ / n
    end

    layer.weight_delta = Δ
    layer.data_layer.weights += Δ
end


function get_pre_activation(layer::NeuralLayer, input::InputTensor)
    layer.weights' * vectorized_data(input)
end


function get_grad_weights{T<:NeuralLayer}(
    layer::LinkedLayer{T},
    error_signal::T_2D_TENSOR
)
    batch_size = size(error_signal)[2]
    vectorized_data(layer.input) * error_signal' / batch_size
end


function get_grad_error_wrt_net{T<:NeuralLayer}(
    layer::LinkedLayer{T},
    error_signal::T_2D_TENSOR
)
    ∇activate = get_∇activator(layer.prev.data_layer.activator)
    weights = layer.data_layer.weights
    prev_pre_activation = vectorized_data(InputTensor(layer.prev.pre_activation))
    (weights * error_signal) .* ∇activate(prev_pre_activation)
end


function format_error_signal(layer::NeuralLayer, error_signal::T_TENSOR)
    size_error_signal = size(error_signal)
    error_signal_dims = length(size_error_signal)

    if error_signal_dims == 4 && length(size(layer.prev.data_layer.weights)) == 2
        vectorized_data(InputTensor(error_signal))
    else
        error_signal
    end
end
