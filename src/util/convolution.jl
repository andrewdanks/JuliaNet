@pyimport scipy.signal as signal

function convn_valid(A::T_TENSOR, B::T_TENSOR)
    signal.convolve(A, B, "valid")
end


function convn_full(A::T_TENSOR, B::T_TENSOR)
    signal.convolve(A, B, "full")
end

function conv2_valid(A::T_TENSOR, B::T_TENSOR)
    signal.convolve2d(A, B, "valid")
end


function conv2_full(A::T_TENSOR, B::T_TENSOR)
    signal.convolve2d(A, B, "full")
end