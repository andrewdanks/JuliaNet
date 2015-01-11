using JuliaNet
using Base.Test

# 1d tests
M = rand(784)
input_tensor = JuliaNet.InputTensor(M)

@test M == squeeze(JuliaNet.vectorized_data(input_tensor), 2)
@test M == squeeze(JuliaNet.input_data(input_tensor, 1), (1, 2))
@test M == squeeze(JuliaNet.map_data(input_tensor, 1), (1, 2))
@test M == squeeze(JuliaNet.input_map_data(input_tensor, 1, 1), 1)

# 2d tests
M = rand(784, 100)
input_tensor = JuliaNet.InputTensor(M)

@test M == JuliaNet.vectorized_data(input_tensor)
@test squeeze(M[:, 99], 2) == squeeze(JuliaNet.input_data(input_tensor, 99), (1, 2))
@test M == squeeze(JuliaNet.map_data(input_tensor, 1), 2)'
@test squeeze(M[:, 99], 2) == squeeze(JuliaNet.input_map_data(input_tensor, 99, 1), 1)

# 3d tests
M = rand(100, 3, 784)

input_tensor = JuliaNet.InputTensor(M)
@test squeeze(M[99, :, :], 1) == squeeze(JuliaNet.input_data(input_tensor, 99), 3)
@test squeeze(M[:, 2, :], 2) == squeeze(JuliaNet.map_data(input_tensor, 2), 3)
@test squeeze(M[99, 2, :], (1, 2)) == squeeze(JuliaNet.input_map_data(input_tensor, 99, 2), 2)

input_tensor = JuliaNet.InputTensor(M, (28, 28))
@test reshape(squeeze(M[99, :, :], 1), 3, 28, 28) == JuliaNet.input_data(input_tensor, 99)
@test reshape(squeeze(M[:, 2, :], 2), 100, 28, 28) == JuliaNet.map_data(input_tensor, 2)
@test reshape(squeeze(M[99, 2, :], (1, 2)), 28, 28) == JuliaNet.input_map_data(input_tensor, 99, 2)

#4d tests
M = rand(100, 3, 28, 28)
input_tensor = JuliaNet.InputTensor(M)
@test squeeze(M[99, :, :, :], 1) == JuliaNet.input_data(input_tensor, 99)
@test squeeze(M[:, 2, :, :], 2) == JuliaNet.map_data(input_tensor, 2)
@test squeeze(M[99, 2, :, :], (1, 2)) == JuliaNet.input_map_data(input_tensor, 99, 2)
