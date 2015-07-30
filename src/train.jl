function fit!(
    nn::NeuralNetwork,
    params::HyperParams,
    training_batches::Vector{Batch},
    validation_batch::Batch=None;

    verbose::Bool=true,
    parallelize::Bool=false,
    params_update_fn=None
)
    if verbose
        show(params)
    end

    batch_chunks = Vector[]
    if parallelize
        batch_chunks = get_batch_chunks(training_batches)
    end
    
    history = FitHistory()
    for current_epoch = 1:params.epochs
        if parallelize
            training_loss, training_classification_error = fit_epoch_parallel!(nn, params, batch_chunks)
        else
            training_loss, training_classification_error = fit_epoch!(nn, params, training_batches)
        end

        if training_classification_error != None
            record_training_history!(history, training_loss, training_classification_error)
        else
            record_training_history!(history, training_loss)
        end

        if validation_batch != None
            fit_validation!(nn, params, validation_batch, history)
        end

        if verbose
            show(history)
        end

        # Determine if we should early stop
        if params.stop_criterion_fn(EarlyStopCriterion(current_epoch, history))
            if verbose
                println("Early stopping after ", current_epoch, " epochs...")
            end
            break 
        end

        if params_update_fn != None
            params = params_update_fn(ParamUpdateCriterion(params, current_epoch, history))
        end
    end
end


function fit_validation!(
    nn::NeuralNetwork,
    params::HyperParams,
    validation_batch::Batch,
    history::FitHistory
)
    valid_output = forward_pass(nn, validation_batch.input)
    validation_loss = params.loss_fn(valid_output, validation_batch.target_output)
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
    batches::Vector{Batch}
)
    linked_layers = LinkedLayer[]
    num_batches = length(batches)

    function do_fit_and_update(batch::Batch)
        # This is a bit of a hack since there's only one property
        # in LinkedLayer that we want to maintain between batches
        linked_layers = get_linked_layers(nn.layers, linked_layers)
        results = fit_batch(nn, linked_layers, params, batch)
        for (idx, layer) in enumerate(linked_layers)
            update_weights!(layer, params)
        end
        return results
    end

    batch_results = reduce(+, [do_fit_and_update(batch) for batch in batches])

    if isdefined(batch_results, :classification_error)
        (batch_results.loss, batch_results.classification_error / num_batches)
    else
        batch_results.loss, None
    end
end


function fit_epoch_parallel!(
    nn::NeuralNetwork,
    params::HyperParams,
    batch_chunks::Vector
)
    linked_layers = LinkedLayer[]
    num_batches = length(batch_chunks) * 2

    function do_fit_batch(batch::Batch)
        # This is a bit of a hack since there's only one property
        # in LinkedLayer that we want to maintain between batches
        linked_layers = get_linked_layers(nn.layers, linked_layers)
        fit_batch(nn, linked_layers, params, batch)
    end

    all_batch_results = BatchResults[]
    for (idx, chunks) in enumerate(batch_chunks)
        batch_results = mapreduce(
            ref -> fetch(ref), +,
            [@spawn do_fit_batch(batch_chunk) for batch_chunk in chunks]
        )
        for (idx, layer) in enumerate(linked_layers)
            layer.grad_weights = batch_results.grad_weights[idx]
            update_weights!(layer, params)
        end
        push!(all_batch_results, batch_results)
    end
    batch_results = reduce(+, all_batch_results)

    if isdefined(batch_results, :classification_error)
        (batch_results.loss, batch_results.classification_error / num_batches)
    else
        batch_results.loss, None
    end
end


function fit_batch(
    nn::NeuralNetwork,
    linked_layers::Vector{LinkedLayer},
    params::HyperParams,
    batch::Batch
)
    # Push the batch through the network and get the output and error
    output = forward_pass!(batch.input, linked_layers[1])

    # Propagate the error signal back through the network and calculate the gradients
    error_signal = -linked_layers[end].data_layer.activator.grad_activation_fn(
        output, batch.target_output
    )
    backward_pass!(error_signal, linked_layers[end])

    grad_weights = [layer.grad_weights for layer in linked_layers]
    loss = params.loss_fn(output, batch.target_output)

    if isdefined(batch, :target_classes)
        classification_error = test_error(predict(nn, output), batch.target_classes)
        BatchResults(grad_weights, loss, classification_error)
    else
        BatchResults(grad_weights, loss)
    end
end


function forward_pass!(
    input::InputTensor,
    layer::LinkedLayer
)
    layer.input = input
    layer.pre_activation = get_pre_activation(layer.data_layer, input)
    activation = InputTensor(
        layer.data_layer.activator.activation_fn(layer.pre_activation)
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
        if isdefined(layer, :dropout_coefficient) && layer.dropout_coefficient > 0
            pre_activation = pre_activation .* (1 - layer.dropout_coefficient)
        end

        activation = layer.activator.activation_fn(pre_activation)
        input = InputTensor(activation)
    end
    vectorized_data(input)
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
