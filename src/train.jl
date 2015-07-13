function fit!(
    nn::NeuralNetwork,
    train_data::Matrix{T_FLOAT},
    targets::Vector{T_FLOAT},
    params::HyperParams;
    
    valid_data::Matrix{T_FLOAT}=None,
    valid_targets::Vector{T_FLOAT}=None,

    verbose::Bool=true
)
    params_update_fn = None

    if verbose
        show(params)
    end

    num_inputs = num_data_matrix_inputs(train_data)

    @assert num_inputs % params.batch_size == 0 "Bad batch size for this data set size"
    num_batches = int(num_inputs / params.batch_size)
    
    nn.classes = sort(unique(targets))
    num_classes = length(nn.classes)

    # Split up the training data into batches
    batches = get_batches(
        nn.classes, num_batches, params.batch_size, train_data, targets
    )
    # Free memory
    train_data, targets = 0, 0; gc()
    
    training_loss_history = T_FLOAT[]
    training_classifcation_error_history = T_FLOAT[]
    validation_loss_history = T_FLOAT[]
    validation_classifcation_error_history = T_FLOAT[]

    do_calculate_validation_loss = valid_data != None && valid_targets != None
    if do_calculate_validation_loss
        valid_data = InputTensor(valid_data)
        valid_target_output = get_target_output_matrix(nn.classes, valid_targets)
    end

    start_time = time()
    for current_epoch = 1:params.epochs
        if verbose
            println("==== Epoch ", current_epoch, " ====")
        end

        training_loss, training_classification_error = fit_epoch!(nn, params, batches, false)

        push!(training_loss_history, training_loss)
        push!(training_classifcation_error_history, training_classification_error)

        if do_calculate_validation_loss
            valid_output = forward_pass(nn, valid_data)
            validation_loss = params.loss_fn(valid_output, valid_target_output)
            validation_classifcation_error = test_error(predict(nn, valid_output), valid_targets)
            push!(validation_loss_history, validation_loss)
            push!(validation_classifcation_error_history, validation_classifcation_error)
        else
            validation_loss = 0.
            validation_classifcation_error = 0.
        end

        if verbose
            println(
                "Training Loss:\t\t\t",
                round(training_loss, 5)
            )
            println(
                "Training Classifcation Error:\t",
                round(training_classification_error, 5)
            )
            if do_calculate_validation_loss
                println(
                    "Validation Loss:\t\t",
                    round(validation_loss, 5)
                )
                println(
                    "Validation Classifcation Error:\t",
                    round(validation_classifcation_error, 5)
                )
            end
        end

        # Determine if we should early stop
        if params.stop_criterion_fn(EarlyStopCriterion(
            current_epoch, training_loss_history, validation_loss_history
        ))
            if verbose
                println("Early stopping after ", current_epoch, " epochs...")
            end
            break 
        end

        if params_update_fn != None
            params = params_update_fn(ParamUpdateCriterion(
                params, current_epoch, training_loss_history, validation_loss_history
            ))
        end
    end
    
    finish_time = time()
    total_train_time = finish_time - start_time
    if verbose
        println("Took ", round(total_train_time / 60, 1), " minutes to fit")
    end

    FitResults(
        training_loss_history,
        validation_loss_history,
        total_train_time
    )
end


function fit_epoch!(
    nn::NeuralNetwork,
    params::HyperParams,
    batches::Vector{Batch},
    parallelize::Bool
)
    linked_layers = LinkedLayer[]
    function call_func(batch::Batch)
        # This is a bit of a hack since there's only one property
        # in LinkedLayer that we want to maintain between batches
        linked_layers = get_linked_layers(nn.layers, linked_layers)
        fit_batch(nn, linked_layers, params, batch)
    end

    if parallelize
        batch_results = mapreduce(
            ref -> fetch(ref),
            +, 
            [@spawn call_func(batch) for batch in batches]
        )
    else
        batch_results = reduce(
            +, [call_func(batch) for batch in batches]
        )
    end

    (
        batch_results.loss,
        batch_results.classification_error / length(batches)
    )
end


function fit_batch(
    nn::NeuralNetwork,
    linked_layers::Vector{LinkedLayer},
    params::HyperParams,
    batch::Batch
)
    # Push the batch through the network and get the output and error
    output = forward_pass!(batch.input, linked_layers[1])

    # Propagate the error signal back through the work and calculate the gradients
    error_signal = -linked_layers[end].activator.grad_activation_fn(
        output, batch.target_output
    )
    backward_pass!(error_signal, linked_layers[end])

    grad_weights = [layer.grad_weights for layer in linked_layers]
    loss = params.loss_fn(output, batch.target_output)
    classification_error = test_error(predict(nn, output), batch.target_classes)

    for (idx, layer) in enumerate(nn.layers)
        if params.momentum > 0 && isdefined(linked_layers[idx], :prev_weight_delta)
            prev_weight_delta = linked_layers[idx].prev_weight_delta 
        else
            prev_weight_delta = 0
        end
        weight_delta = params.momentum * prev_weight_delta - params.learning_rate * grad_weights[idx]
        linked_layers[idx].weight_delta = weight_delta
        layer.weights += weight_delta
    end

    BatchResults(grad_weights, loss, classification_error)
end


function get_pre_activation(weights::Matrix, input::InputTensor)
    weights' * vectorized_data(input)
end


function get_activation(activation_fn::Function, pre_activation::T_TENSOR)
    InputTensor(activation_fn(pre_activation))
end


function forward_pass!(
    input::InputTensor,
    layer::LinkedLayer
)
    layer.input = input
    layer.pre_activation = get_pre_activation(layer.weights, input)
    layer.activation = get_activation(layer.activator.activation_fn, layer.pre_activation)
    if has_next(layer)
        forward_pass!(layer.activation, layer.next)
    else
        vectorized_data(layer.activation)
    end
end


function backward_pass!(
    error_signal::T_TENSOR,
    layer::LinkedLayer
)
    layer.grad_weights = get_grad_weights(layer.input, error_signal)
    if has_prev(layer)
        prev_pre_activation = layer.prev.pre_activation
        backward_pass!(
            get_grad_error_wrt_net(
                layer.prev.activator.grad_activation_fn,
                layer.weights,
                layer.prev.pre_activation,
                error_signal
            ),
            layer.prev
        )
    end
end


function forward_pass(nn::NeuralNetwork, input::InputTensor)
    for layer in nn.layers
        pre_activation = get_pre_activation(layer.weights, input)
        input = get_activation(layer.activator.activation_fn, pre_activation)
    end
    vectorized_data(input)
end


function get_grad_weights(input::InputTensor, error_signal::T_TENSOR)
    batch_size = size(error_signal)[2]
    vectorized_data(input) * error_signal' / batch_size
end


function get_grad_error_wrt_net(
    grad_activation_fn::Function,
    weights::Matrix,
    prev_pre_activation::T_TENSOR,
    error_signal::T_2D_TENSOR
)
    (weights * error_signal) .* grad_activation_fn(prev_pre_activation)
end
