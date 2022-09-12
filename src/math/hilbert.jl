export encode_hilbert

const HILBERT_SIZE = 2^16


# Normalizes the coordinate to the range [0, 65535] (2^16 - 1)
# This guarantees that the Morton code will fit in 32 bits, since
# (2^16 - 1)^2 < 2^32 - 1
function encode_hilbert(x::T, y::T, s_inv::T) where {T <: Union{Float32, Float64}}
    x_u32 = floor(UInt32, x * s_inv * (HILBERT_SIZE - 1))
    y_u32 = floor(UInt32, y * s_inv * (HILBERT_SIZE - 1))
    return encode_hilbert(x_u32, y_u32)
end

# Encodes the coordinates into a Hilbert index by
# reinterpretting the bits as a 32-bit unsigned integer.
function encode_hilbert(x::Float32, y::Float32)
    x_u64 = widen(reinterpret(UInt32, x))
    y_u64 = widen(reinterpret(UInt32, y))
    return encode_hilbert(x_u64, y_u64)
end

function encode_hilbert(x::Float64, y::Float64)
    # Convert to a 32-bit float first.
    x_u64 = widen(reinterpret(UInt32, Float32(x)))
    y_u64 = widen(reinterpret(UInt32, Float32(y)))
    return encode_hilbert(x_u64, y_u64)
end

# For UInt32, assumes x, y are in the range [0, 2^16)
# For UInt64, assumes x, y are in the range [0, 2^32)
function encode_hilbert(x::T, y::T) where {T <: Union{UInt32, UInt64}}
    # n is the max side length of the Hilbert curve (2^16 or 2^32)
    # to be able to store the index in a single integer of the same type
    if T === UInt32
        n = T(2^16)
    elseif T === UInt64
        n = T(2^32)
    else
        throw(ArgumentError("Invalid type for x, y: $T"))
    end
    d = T(0)
    s = n >> T(1)
    while s > T(0)
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
