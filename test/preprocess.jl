using JuliaNet
using Base.Test

R = rand(10, 10)

@test mean(zero_mean(R), 2) == 0
@test var(unit_variance(R), 2) == 1
