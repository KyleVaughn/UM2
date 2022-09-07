export encode_hilbert

# Side length of the square. 
# Needs to be a power of 2, at least 2, and n^2 < 2^31, unless you want to use Int64.
const HILBERT_SIZE = 2^15

function encode_hilbert(x::T, y::T, s_inv::T) where {T <: Union{Float32, Float64}}
    x_i32 = floor(Int32, x * s_inv * HILBERT_SIZE)
    y_i32 = floor(Int32, y * s_inv * HILBERT_SIZE)
    if x_i32 === Int32(2^15)
        x_i32 -= Int32(1)
    end
    if y_i32 === Int32(2^15)
        y_i32 -= Int32(1)
    end
    return encode_hilbert(x_i32, y_i32)
end

# Valid for Int32, Int64
function encode_hilbert(x::T, y::T) where {T <: Union{Int32, Int64}}
    n = T(HILBERT_SIZE)
    # Assumes x and y are in the range [0, n - 1]
    d = T(0)
    s = n >> T(1)
    while s > (0)
        rx = T((x & s) > T(0))
        ry = T((y & s) > T(0))
        d += s * s * ((T(3) * rx) ⊻ ry);
        if ry === T(0)
            if rx === T(1) 
                x = n - T(1) - x
                y = n - T(1) - y
            end
            # Swap x and y
            x, y = y, x
        end
        s >>= T(1)
    end
    return d
end
