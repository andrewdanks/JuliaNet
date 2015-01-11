function fit!(
    nn::NeuralNetwork,
    
    train_data::T_TENSOR,
    targets::Vector{T_FLOAT},
    
    params::HyperParams=HyperParams();
    
    valid_data=None,
    valid_targets=None,

    verbose::Bool=true,
    
    # Function that gets called at the end of each epoch to update
    # hyper params (such as learning rate and momentum)
    params_update_fn!::Function=DEFAULT_PARAMS_UPDATE_FN,

    loss_fn::Function=mean_squared_error
)
    if verbose
        show(params)
    end

    # TODO: this function needs to be refactored to handle 4d inputs

    num_inputs = num_data_matrix_inputs(train_data)

    # Ensure we have a valid batch size
    assert(num_inputs % params.batch_size == 0)
    num_batches = int(num_inputs / params.batch_size)
    
    nn.classes = sort(unique(targets))
    num_classes = length(nn.classes)

    # Split up the training data into batches
    batches = get_batches(nn.classes, num_batches, params.batch_size, train_data, targets)
    # Free memory
    train_data, targets = 0, 0; gc()

    batches = format_batches_for_input_layer(input_layer(nn), batches)
    
    training_loss_history = T_FLOAT[]
    training_classifcation_error_history = T_FLOAT[]
    validation_loss_history = T_FLOAT[]
    validation_classifcation_error_history = T_FLOAT[]

    do_calculate_validation_loss = valid_data != None && valid_targets != None
    if do_calculate_validation_loss
        valid_data = format_input(input_layer(nn), valid_data)
        valid_target_output = target_output_matrix(nn.classes, valid_targets)
    end

    start_time = time()  # not the most accurate measure
    for current_epoch = 1:params.epochs

        if verbose
            println("==== Epoch ", current_epoch, " ====")
        end

        batch_training_loss_history = T_FLOAT[]
        batch_training_classifcation_error_history = T_FLOAT[]

        for i = 1:num_batches
            batch, batch_target_classes, batch_target_output = batches[i]

            # Push the batch through the network and get the output and error
            batch_output = forward_pass!(nn, params, batch)

            # Calculate the losses:
            batch_training_loss = loss_fn(batch_output, batch_target_output)
            batch_training_classification_error = test_error(predict(nn, batch_output), batch_target_classes)
            push!(batch_training_loss_history, batch_training_loss)
            push!(batch_training_classifcation_error_history, batch_training_classification_error)
            
            # Propagate the error signal back through the work and calculate the gradients
            backward_pass!(nn, params, batch_output, batch_target_output)

            # Update the weights according to the gradients
            update_weights!(nn, params)
        end

        # Calculate the total losses
        training_loss = sum(batch_training_loss_history)
        training_classification_error = mean(batch_training_classifcation_error_history)
        push!(training_loss_history, training_loss)
        push!(training_classifcation_error_history, training_classification_error)
        if do_calculate_validation_loss
            valid_output = forward_pass(nn, valid_data)
            validation_loss = loss_fn(valid_output, valid_target_output)
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
        if params.stop_criterion_fn(
            current_epoch,
            training_loss,
            validation_loss,
            training_loss_history,
            validation_loss_history
        )
            if verbose
                println("Early stopping after ", current_epoch, " epochs...")
            end
            break 
        end

        params_update_fn!(params, e, training_loss_history, validation_loss_history)
    end
    
    finish_time = time()
    total_train_time = finish_time - start_time
    if verbose
        println("Took ", round(total_train_time / 60, 1), " minutes to train")
    end

    deactivate_network!(nn)

    FitResults(
        training_loss_history,
        validation_loss_history,
        total_train_time
    )
end

function target_output_matrix(
    classes::Vector,
    target_classes::Vector
)
    num_classes = length(classes)
    output_size = length(target_classes)
    target_output = zeros(num_classes, output_size)
    for j = 1:output_size
        target = zeros(num_classes)
        target[findfirst(classes, target_classes[j])] = 1.
        target_output[:, j] = target
    end
    target_output
end

function get_batches(
    classes::Vector,
    num_batches::T_INT,
    batch_size::T_INT,
    data::Matrix{T_FLOAT},
    target_classes::Vector
)
    batch_target_output_by_batch_number = Dict()
    for i = 1:num_batches
        batch = get_ith_batch(data, i, batch_size)
        batch_target_classes = get_ith_batch(target_classes, i, batch_size)
        batch_target_output = target_output_matrix(classes, batch_target_classes)
        batch_target_output_by_batch_number[i] = batch, batch_target_classes, batch_target_output
    end
    batch_target_output_by_batch_number
end

function format_input(input_layer::InputLayer, input::T_TENSOR)
    reshape_dims = input_layer.feature_map_size
    InputTensor(input, reshape_dims)
end

function format_batches_for_input_layer(input_layer::InputLayer, batches::Dict)
    num_batches = length(batches)
    new_batches = Dict()
    for i = 1:num_batches
        batch, x, y = batches[i]
        new_batch = format_input(input_layer, batch)
        new_batches[i] = (new_batch, x, y)
    end
    new_batches
end

function forward_pass!(nn::NeuralNetwork, params::HyperParams, input::InputTensor)
    for layer in all_layers(nn)
        activate_layer!(layer, input)
        input = layer.activation
    end
    vectorized_data(output_layer(nn).activation)
end

function forward_pass(nn::NeuralNetwork, input::InputTensor)
    for layer in all_layers(nn)
        test_activate_layer!(layer, input)
        input = layer.activation
    end
    output = output_layer(nn).activation
    deactivate_network!(nn)
    vectorized_data(output)
end

# function accelerate_gradient!(layer::NeuralLayer, params::HyperParams)
#     layer.weight_delta += params.momentum .* layer.prev_weight_delta
# end

function backward_pass!(
    nn::NeuralNetwork,
    params::HyperParams,
    output::T_2D_TENSOR,
    target_output::T_2D_TENSOR
)
    error_signal = -output_layer(nn).activator.grad_activation_fn(output, target_output)

    layers = reverse(all_layers(nn))
    num_layers = length(layers)

    for (i, layer) in enumerate(layers)
        if i < num_layers
            if i + 1 == num_layers
                next_layer = None
            else
                next_layer = layers[i + 1]
            end
            error_signal = backward_pass!(layer, params, error_signal, next_layer)
        end
    end
end