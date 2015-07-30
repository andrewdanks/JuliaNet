using JuliaNet
using Base.Test

THRESHOLD = 1e-10

M = rand(100, 28, 28)
k = rand(1, 5, 5)

konvn_valid = JuliaNet.convn_valid(M, k)

konv2_valid = zeros(100, 25, 25)
for i = 1:100
    incr =  JuliaNet.conv2_valid(squeeze(M[i, :, :], 1), squeeze(k, 1))
    konv2[i, :, :] = squeeze(konv2[i, :, :], 1) + incr
end

num_not_equal = length(find(x -> x < THRESHOLD, konvn_valid - konv2))
@test num_not_equal == 0