type HyperParams
    learning_rate::T_FLOAT
    adaptive_learning_rates::Bool

    momentum::T_FLOAT
    accelerated_gradient::Bool

    L1_decay::T_FLOAT
    L2_decay::T_FLOAT

    epochs::T_INT
    batch_size::T_INT

    stop_criterion_fn::Function

    rmsprop::Bool

    function HyperParams(
        learning_rate::T_FLOAT=1.,
        
        # Number of times to sweep through the data
        epochs::T_INT=1,

        # How much data to process before each weight update
        # 1 = online learning, Inf=full batch
        batch_size::T_INT=1

        ;

        momentum::T_FLOAT=0.,

        # Nesterov’s Accelerated Gradient
        accelerated_gradient::Bool=false,

        adaptive_learning_rates::Bool=false, 

        L1_decay::T_FLOAT=0.,
        L2_decay::T_FLOAT=0.,

        # Automatically stop training based on some criterion
        stop_criterion_fn::Function=no_stop_criterion_fn,

        rmsprop::Bool=false
    )

        assert(batch_size > 0)

        new(
            learning_rate,
            adaptive_learning_rates,
            momentum,
            accelerated_gradient,
            L1_decay,
            L2_decay,
            epochs,
            batch_size,
            stop_criterion_fn,
            rmsprop
        )
    end
end

function Base.show(io::IO, params::HyperParams)
    println(io, "Learning Rate: ", params.learning_rate)
    #println(io, "Adaptive Leanring Rates: ", params.adaptive_learning_rates)
    println(io, "Momentum: ", params.momentum)
    #println(io, "Accelerated: ", params.accelerated_gradient)
    #println(io, "L1 decay: ", params.L1_decay)
    #println(io, "L2 decay: ", params.L2_decay)
    println(io, "Epochs: ", params.epochs)
    println(io, "Batch Size: ", params.batch_size)
    println(io, "Stop Criterion: ", params.stop_criterion_fn)
    #println(io, "rmsprop: ", params.rmsprop)
end
