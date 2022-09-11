export encode_morton


const MAX_MORTON_INDEX = 0x000000000000ffff # 2^16 - 1

# Normalizes the coordinate to the range [0, 65535] (2^16 - 1)
# This guarantees that the Morton code will fit in 32 bits, since
# (2^16 - 1)^2 < 2^32 - 1
function encode_morton(x::T, y::T, s_inv::T) where {T <: Union{Float32, Float64}}
    x_u32 = floor(UInt32, x * s_inv * MAX_MORTON_INDEX)
    y_u32 = floor(UInt32, y * s_inv * MAX_MORTON_INDEX)
    return encode_morton(x_u32, y_u32)
end

# This doesn't produce a good correlation between space and morton index.
## Encodes the coordinates into a Morton number by
## reinterpretting the bits as a 32-bit unsigned integer.
#function encode_morton(x::Float32, y::Float32)
#    x_u64 = widen(reinterpret(UInt32, x))
#    y_u64 = widen(reinterpret(UInt32, y))
#    return encode_morton(x_u64, y_u64)
#end
#
#function encode_morton(x::Float64, y::Float64)
#    # Convert to a 32-bit float first.
#    x_u64 = widen(reinterpret(UInt32, Float32(x)))
#    y_u64 = widen(reinterpret(UInt32, Float32(y)))
#    return encode_morton(x_u64, y_u64)
#end

# For UInt32, assumes x, y are in the range [0, 2^16)
# For UInt64, assumes x, y are in the range [0, 2^32)
if UM2_HAS_BMI2
    function encode_morton(x::UInt32, y::UInt32)
        return pdep(x, 0x55555555) | pdep(y, 0xaaaaaaaa)
    end
    function encode_morton(x::UInt64, y::UInt64)
        return pdep(x, 0x5555555555555555) | pdep(y, 0xaaaaaaaaaaaaaaaa)
    end
else
    function encode_morton(x::UInt32, y::UInt32)
        x = (x | (x << 16)) & 0x0000ffff
        x = (x | (x << 8))  & 0x00ff00ff
        x = (x | (x << 4))  & 0x0f0f0f0f
        x = (x | (x << 2))  & 0x33333333
        x = (x | (x << 1))  & 0x55555555
    
        y = (y | (y << 16)) & 0x0000ffff
        y = (y | (y << 8))  & 0x00ff00ff
        y = (y | (y << 4))  & 0x0f0f0f0f
        y = (y | (y << 2))  & 0x33333333
        y = (y | (y << 1))  & 0x55555555

        return x | (y << 1)
    end
    function encode_morton(x::UInt64, y::UInt64)
        x = (x | (x << 32)) & 0x00000000ffffffff
        x = (x | (x << 16)) & 0x0000ffff0000ffff
        x = (x | (x << 8))  & 0x00ff00ff00ff00ff
        x = (x | (x << 4))  & 0x0f0f0f0f0f0f0f0f
        x = (x | (x << 2))  & 0x3333333333333333
        x = (x | (x << 1))  & 0x5555555555555555
    
        y = (y | (y << 32)) & 0x00000000ffffffff
        y = (y | (y << 16)) & 0x0000ffff0000ffff
        y = (y | (y << 8))  & 0x00ff00ff00ff00ff
        y = (y | (y << 4))  & 0x0f0f0f0f0f0f0f0f
        y = (y | (y << 2))  & 0x3333333333333333
        y = (y | (y << 1))  & 0x5555555555555555
    
        return x | (y << 1)
    end
end
