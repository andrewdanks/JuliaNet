type HyperParams
    learning_rate::T_FLOAT
    momentum::T_FLOAT
    epochs::T_INT
    batch_size::T_INT
    stop_criterion_fn::Function
    loss_fn::Function

    function HyperParams(
        learning_rate::T_FLOAT=1.,
        # Number of times to sweep through the data
        epochs::T_UINT=1,

        # How much data to process before each weight update
        # 1 = online learning, Inf=full batch
        batch_size::T_UINT=1

        ;

        momentum::T_FLOAT=0.,

        # Automatically stop training based on some criterion
        stop_criterion_fn::Function=no_stop_criterion_fn,

        loss_fn::Function=mean_squared_error
    )

        assert(batch_size > 0)

        new(
            learning_rate,
            momentum,
            epochs,
            batch_size,
            stop_criterion_fn,
            loss_fn
        )
    end
end

function Base.show(io::IO, params::HyperParams)
    println(io, "Learning Rate: ", params.learning_rate)
    println(io, "Momentum: ", params.momentum)
    println(io, "Epochs: ", params.epochs)
    println(io, "Batch Size: ", params.batch_size)
    println(io, "Stop Criterion: ", params.stop_criterion_fn)
end
