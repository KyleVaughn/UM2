const HILBERT_SIZE = 2^10

function encode_hilbert(x::T, y::T, s_inv::T) where {T <: AbstractFloat}
    # Assumes x and y are in the range [0, s]
    # s_inv is the inverse of the side length of the square
    # x and y are the coordinates of the point

    # Convert to UInt16
    x_u16 = floor(Int16, x * s_inv * HILBERT_SIZE)
    y_u16 = floor(Int16, y * s_inv * HILBERT_SIZE)
    return encode_hilbert(x_u16, y_u16)
end

function encode_hilbert(x::T, y::T) where {T <: Integer}
    n = T(HILBERT_SIZE)
    # Assumes x and y are in the range [0, n - 1]
    d = T(0)
    s = n >> T(1)
    while s > 0
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
