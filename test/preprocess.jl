using JuliaNet
using Base.Test

R = rand(10, 10)
EPSILON = 10E-8

@test all(mean(zero_mean(R), 2) .< EPSILON)
@test all(var(unit_variance(R), 2) .- 1 .< EPSILON)
