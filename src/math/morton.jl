export morton_encode, 
       morton_decode,
       normalized_morton_encode,
       normalized_morton_decode
#
# Routines to support Morton ordered (Z-ordered) data 
#
# 32-bit or 64-bit unsigned integer
const MortonCode = UInt32
const MORTON_MAX_INDEX = typemax(MortonCode)
const MORTON_MAX_DEPTH = MortonCode(4 * sizeof(MortonCode))
const MORTON_MAX_SIDE  = MortonCode((1 << MORTON_MAX_DEPTH) - 1)

# For UInt32, assumes x, y are in the range [0, 2^16)
# For UInt64, assumes x, y are in the range [0, 2^32)
if UM_HAS_BMI2
    # Use the BMI2 instructions pdep and pext to encode/decode morton codes
    function morton_encode(x::UInt32, y::UInt32)
        return pdep(x, 0x55555555) | pdep(y, 0xaaaaaaaa)
    end

    function morton_encode(x::UInt64, y::UInt64)
        return pdep(x, 0x5555555555555555) | pdep(y, 0xaaaaaaaaaaaaaaaa)
    end

    morton_decode(z::UInt32) = pext(z, 0x55555555), pext(z, 0xaaaaaaaa)
    morton_decode(z::UInt64) = pext(z, 0x5555555555555555), pext(z, 0xaaaaaaaaaaaaaaaa)
else
    # Equivalent to pdep(x, 0x55555555)
    function pdep_0x55555555(x::UInt32)
        # x <= 0x0000ffff
        x = (x | (x << 8)) & 0x00ff00ff    
        x = (x | (x << 4)) & 0x0f0f0f0f    
        x = (x | (x << 2)) & 0x33333333    
        x = (x | (x << 1)) & 0x55555555
        return x
    end
    
    function pdep_0x5555555555555555(x::UInt64)
        # x <= 0x00000000ffffffff
        x = (x | (x << 16)) & 0x0000ffff0000ffff    
        x = (x | (x <<  8)) & 0x00ff00ff00ff00ff    
        x = (x | (x <<  4)) & 0x0f0f0f0f0f0f0f0f    
        x = (x | (x <<  2)) & 0x3333333333333333    
        x = (x | (x <<  1)) & 0x5555555555555555
        return x
    end
    
    # Equivalent to pext(x, 0x55555555)
    function pext_0x55555555(x::UInt32)
        x &= 0x55555555
        x = (x ^ (x >> 1)) & 0x33333333
        x = (x ^ (x >> 2)) & 0x0f0f0f0f
        x = (x ^ (x >> 4)) & 0x00ff00ff
        x = (x ^ (x >> 8)) & 0x0000ffff
        return x
    end
    
    function pext_0x5555555555555555(x::UInt64)
        x &= 0x5555555555555555;
        x = (x ^ (x >>  1)) & 0x3333333333333333
        x = (x ^ (x >>  2)) & 0x0f0f0f0f0f0f0f0f
        x = (x ^ (x >>  4)) & 0x00ff00ff00ff00ff
        x = (x ^ (x >>  8)) & 0x0000ffff0000ffff
        x = (x ^ (x >> 16)) & 0x00000000ffffffff
        return x
    end

    function morton_encode(x::UInt32, y::UInt32)
        x = pdep_0x55555555(x)
        y = pdep_0x55555555(y)
        return x | (y << 1)
    end

    function morton_encode(x::UInt64, y::UInt64)
        x = pdep_0x5555555555555555(x)
        y = pdep_0x5555555555555555(y)
        return x | (y << 1)
    end

    function morton_decode(z::UInt32)
        x = pext_0x55555555(z)
        y = pext_0x55555555(z >> 1)
        return x, y
    end

    function morton_decode(z::UInt64)
        x = pext_0x5555555555555555(z)
        y = pext_0x5555555555555555(z >> 1)
        return x, y
    end
end

# Normalizes the coordinate to the range [0, MORTON_MAX_SIDE] before computing
# the morton index. This allows for a more sane Float encoding, especially for a known
# domain size.
function normalized_morton_encode(x::T, y::T, scale_inv::T) where {T <: AbstractFloat} 
    x_u = floor(MortonCode, x * scale_inv * MORTON_MAX_SIDE )
    y_u = floor(MortonCode, y * scale_inv * MORTON_MAX_SIDE )
    return morton_encode(x_u, y_u)
end

# Decodes a floating point x,y pair, encoded by "normalized_morton_encode", scaling it
# back to the original domain.
function normalized_morton_decode(z::MortonCode, scale::T) where {T <: AbstractFloat} 
    mms_inv = T(1) / MORTON_MAX_SIDE
    x, y = morton_decode(z)
    return x * scale * mms_inv, y * scale * mms_inv
end
