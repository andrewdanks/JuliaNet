using HDF5, JLD


function fit!(
    nn::NeuralNetwork,
    params::HyperParams,
    training_batches::Vector{Batch},
    validation_batch=T_NONE,
    config::FitConfig=FitConfig();
    loss_fn=mean_squared_error,
    stop_criterion_fn=nothing,
    params_update_fn=nothing,
    history_watcher_fn=nothing
)
    if config.verbose
        show(params)
    end

    batch_chunks = Vector[]
    if config.parallelize
        batch_chunks = get_batch_chunks(training_batches)
    end
    
    history = FitHistory()
    for current_epoch = 1:params.epochs
        if config.parallelize
            training_loss = fit_epoch_parallel!(nn, params, batch_chunks, loss_fn)
        else
            training_loss = fit_epoch!(nn, params, training_batches, loss_fn)
        end
        record_training_history!(history, training_loss)

        if validation_batch != T_NONE
            fit_validation!(nn, params, validation_batch, history, loss_fn)
        end

        if isa(history_watcher_fn, Function)
            history_watcher_fn(history)
        end

        if config.verbose
            show_epoch(STDOUT, history, current_epoch)
        end

        if isa(config.save_file, AbstractString)
            if config.verbose
                print("Saving model to ", config.save_file, "...")
            end
            save(
                config.save_file,
                "model", nn,
                "hyper_params", params,
                "history", history,
                "num_batches", length(training_batches),
                "loss_fn", AbstractString(loss_fn),
                "stop_criterion_fn", AbstractString(stop_criterion_fn),
                "params_update_fn", AbstractString(params_update_fn),
                "JULIANET_VERSION", JULIANET_VERSION
            )
            if config.verbose
                println(" Done.")
            end
        end

        if isa(stop_criterion_fn, Function) && stop_criterion_fn(EarlyStopCriterion(current_epoch, history))
            if config.verbose
                println("Early stopping after ", current_epoch, " epochs...")
            end
            break 
        end

        if isa(params_update_fn, Function)
            params = params_update_fn(ParamUpdateCriterion(params, current_epoch, history))
        end
    end
end


function fit_validation!(
    nn::NeuralNetwork,
    params::HyperParams,
    validation_batch::Batch,
    history::FitHistory,
    loss_fn::Function
)
    valid_output = forward_pass(nn, validation_batch.input)
    validation_loss = loss_fn(valid_output, validation_batch.target_output)
    if isdefined(validation_batch, :target_classes)
        validation_classifcation_error = test_error(predict(nn, valid_output), validation_batch.target_classes)
        record_validation_history!(history, validation_loss, validation_classifcation_error)
    else
        record_validation_history!(history, validation_loss)
    end
end


function fit_epoch!(
    nn::NeuralNetwork,
    params::HyperParams,
    batches::Vector{Batch},
    loss_fn::Function
)
    linked_layers = LinkedLayer[]
    function do_fit_and_update(batch::Batch)
        # This is a bit of a hack since there's only one property
        # in LinkedLayer that we want to maintain between batches
        linked_layers = get_linked_layers(nn.layers, linked_layers)
        output = fit_batch!(linked_layers, batch)
        update_weights!(linked_layers, params)
        loss_fn(output, batch.target_output)
    end

    total_loss = reduce(+, [do_fit_and_update(batch) for batch in batches])
    total_loss / length(batches)
end


function update_weights!(linked_layers::Vector{LinkedLayer}, params::HyperParams)
    for layer in linked_layers
        update_weights!(layer, params)
    end
end


function fit_epoch_parallel!(
    nn::NeuralNetwork,
    params::HyperParams,
    batch_chunks::Vector,
    loss_fn::Function
)
    linked_layers = LinkedLayer[]
    function do_fit_batch(batch::Batch)
        # This is a bit of a hack since there's only one property
        # in LinkedLayer that we want to maintain between batches
        linked_layers = get_linked_layers(nn.layers, linked_layers)
        output = fit_batch!(linked_layers, batch)
        loss_fn(output, batch.target_output)
    end

    total_loss = 0
    for (idx, chunks) in enumerate(batch_chunks)
        loss = mapreduce(
            ref -> fetch(ref), +,
            [@spawn do_fit_batch(batch_chunk) for batch_chunk in chunks]
        )
        update_weights!(linked_layers, params)
        total_loss += loss / length(chunks)
    end
    total_loss / length(batches)
end


function fit_batch!(
    linked_layers::Vector{LinkedLayer},
    batch::Batch
)
    output = forward_pass!(batch.input, linked_layers[1])

    ∇activate = get_∇activator(linked_layers[end].data_layer.activator)
    pre_activation = linked_layers[end].pre_activation
    error_signal = grad_squared_error(output, batch.target_output) .* ∇activate(pre_activation)
    backward_pass!(error_signal, linked_layers[end])

    output
end


function forward_pass!(
    input::InputTensor,
    layer::LinkedLayer
)
    if layer.data_layer.corruption_level > 0
        input = zero_out_with_prob(input, layer.data_layer.corruption_level)
    end

    layer.input = input  # TODO: this is a waste of memory
    layer.pre_activation = get_pre_activation(layer.data_layer, input)
    activation = InputTensor(
        get_activator(layer.data_layer.activator)(layer.pre_activation)
    )
    if layer.data_layer.dropout_coefficient > 0
        activation = zero_out_with_prob(activation, layer.data_layer.dropout_coefficient)
    end
    layer.activation = activation

    if has_next(layer)
        forward_pass!(layer.activation, layer.next)
    else
        vectorized_data(layer.activation)
    end
end


function forward_pass(nn::NeuralNetwork, input::InputTensor)
    for layer in nn.layers
        pre_activation = get_pre_activation(layer, input)
        if layer.dropout_coefficient > 0
            pre_activation = pre_activation .* (1 - layer.dropout_coefficient)
        end

        activation = get_activator(layer.activator)(pre_activation)
        input = InputTensor(activation)
    end
    vectorized_data(input)
end


function forward_pass(nn::NeuralNetwork, batch::Batch)
    forward_pass(nn, batch.input)
end


function backward_pass!(
    error_signal::T_TENSOR,
    layer::LinkedLayer
)
    layer.grad_weights = get_grad_weights(layer, error_signal)
    if has_prev(layer)
        prev_pre_activation = layer.prev.pre_activation
        backward_pass!(
            format_error_signal(
                layer.prev.data_layer,
                get_grad_error_wrt_net(layer, error_signal)
            ),
            layer.prev
        )
    end
end
