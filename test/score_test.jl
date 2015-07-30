using JuliaNet
using Base.Test


nn = NeuralNetwork()
nn.classes = [1,2]

@test predict(nn, [0.3 0.7; 1.0 0]) == [2, 1]

@test_approx_eq test_error([-1,2,3,4,5],[1,-2,3,4,4]) 0.6 
