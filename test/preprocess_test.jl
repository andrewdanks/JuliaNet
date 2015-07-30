using JuliaNet
using Base.Test

EPSILON = 10E-8

@test all(mean(zero_mean(rand(10, 10)), 2) .< EPSILON)

@test all(var(unit_variance(rand(10, 10)), 2) .- 1 .< EPSILON)

@test JuliaNet.get_target_output_matrix(
    [1,2], [1,2,1,1,2,2]
) == [1 0; 0 1; 1 0; 1 0; 0 1; 0 1]'


batch_size = 2
classes = [1,2]
target_classes = [1,2,1,1,2,2,1]

data = float(rand(2, 7))
batches = JuliaNet.get_batches(data, batch_size, classes, target_classes)
@test length(batches) == 4
@test batches[1].input == JuliaNet.InputTensor(data[:, 1:2])
@test batches[1].target_output == [1 0; 0 1]'
@test batches[1].target_classes == [1,2]
@test batches[2].input == JuliaNet.InputTensor(data[:, 3:4])
@test batches[2].target_output == [1 0; 1 0]'
@test batches[2].target_classes == [1,1]
@test batches[3].input == JuliaNet.InputTensor(data[:, 5:6])
@test batches[3].target_output == [0 1; 0 1]'
@test batches[3].target_classes == [2,2]
@test batches[4].input == JuliaNet.InputTensor(data[:, 7])
@test batches[4].target_output == [1 0]'
@test batches[4].target_classes == [1]

data = float(rand(7, 3, 28, 28))
batches = JuliaNet.get_batches(data, batch_size, classes, target_classes)
@test length(batches) == 4
@test batches[1].input == JuliaNet.InputTensor(data[1:2, :, :, :])
@test batches[2].input == JuliaNet.InputTensor(data[3:4, :, :, :])
@test batches[3].input == JuliaNet.InputTensor(data[5:6, :, : ,:])
@test batches[4].input == JuliaNet.InputTensor(data[7, :, :, :])
