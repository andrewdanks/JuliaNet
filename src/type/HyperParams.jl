type HyperParams
    learning_rate::T_FLOAT
    momentum::T_FLOAT
    epochs::T_INT
    stop_criterion_fn::Function
    loss_fn::Function

    function HyperParams(
        learning_rate::T_FLOAT=1.,
        # Number of times to sweep through the data
        epochs::T_UINT=1,

        ;

        momentum::T_FLOAT=0.,

        # Automatically stop training based on some criterion
        stop_criterion_fn::Function=no_stop_criterion_fn,

        loss_fn::Function=cross_entropy_error
    )
        new(
            learning_rate,
            momentum,
            epochs,
            stop_criterion_fn,
            loss_fn
        )
    end
end

function Base.show(io::IO, params::HyperParams)
    println(io, "Learning Rate: ", params.learning_rate)
    println(io, "Momentum: ", params.momentum)
    println(io, "Epochs: ", params.epochs)
    println(io, "Stop Criterion: ", params.stop_criterion_fn)
end
