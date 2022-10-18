export encode_morton, decode_morton, normalized_encode_morton
#
# Routines to support Morton ordered (Z-ordered) data 
#
# 32-bit or 64-bit unsigned integer
const MORTON_INDEX_TYPE = UInt32
# 0xffffffff or 0xffffffffffffffff
const MORTON_MAX_INDEX = typemax(MORTON_INDEX_TYPE)
# 16 or 32
const MORTON_MAX_LEVELS = 4 * sizeof(MORTON_INDEX_TYPE)
# 0x0000ffff or 0x00000000ffffffff
const MORTON_MAX_SIDE  = MORTON_MAX_INDEX >> MORTON_MAX_LEVELS

# For UInt32, assumes x, y are in the range [0, 2^16]
# For UInt64, assumes x, y are in the range [0, 2^32]
if UM_HAS_BMI2
    # Use the BMI2 instructions pdep and pext to encode/decode morton codes
    function encode_morton(x::UInt32, y::UInt32)
        return pdep(x, 0x55555555) | pdep(y, 0xaaaaaaaa)
    end

    function encode_morton(x::UInt64, y::UInt64)
        return pdep(x, 0x5555555555555555) | pdep(y, 0xaaaaaaaaaaaaaaaa)
    end

    decode_morton(z::UInt32) = pext(z, 0x55555555), pext(z, 0xaaaaaaaa)
    decode_morton(z::UInt64) = pext(z, 0x5555555555555555), pext(z, 0xaaaaaaaaaaaaaaaa)
else
    # Magic numbers for interleaving bits
    const morton_masks_32 = (
        0x0000ffff, 
        0x00ff00ff,
        0x0f0f0f0f,
        0x33333333,
        0x55555555,
    )
    
    const morton_masks_64 = (
        0x00000000ffffffff,
        0x0000ffff0000ffff,
        0x00ff00ff00ff00ff,
        0x0f0f0f0f0f0f0f0f,
        0x3333333333333333,
        0x5555555555555555,
    )     

    # Equivalent to pdep(x, 0x55555555)
    function pdep_0x55555555(x::UInt32)
        # First mask is unnecessary, since x is already less then or equal to 2^16
        # x &= morton_masks_32[1]
        x = (x | (x << 8)) & morton_masks_32[2]
        x = (x | (x << 4)) & morton_masks_32[3]
        x = (x | (x << 2)) & morton_masks_32[4]
        x = (x | (x << 1)) & morton_masks_32[5]
        return x
    end
    
    function pdep_0x5555555555555555(x::UInt64)
        # First mask is unnecessary, since x is already less then or equal to 2^32
        # x &= morton_masks_64[1]
        x = (x | (x << 16)) & morton_masks_64[2]
        x = (x | (x <<  8)) & morton_masks_64[3]
        x = (x | (x <<  4)) & morton_masks_64[4]
        x = (x | (x <<  2)) & morton_masks_64[5]
        x = (x | (x <<  1)) & morton_masks_64[6]
        return x
    end
    
    # Equivalent to pext(x, 0x55555555)
    function pext_0x55555555(x::UInt32)
        x &= morton_masks_32[5]
        x = (x ^ (x >> 1)) & morton_masks_32[4]
        x = (x ^ (x >> 2)) & morton_masks_32[3]
        x = (x ^ (x >> 4)) & morton_masks_32[2]
        x = (x ^ (x >> 8)) & morton_masks_32[1]
        return x
    end
    
    function pext_0x5555555555555555(x::UInt64)
        x &= morton_masks_64[6]
        x = (x ^ (x >>  1)) & morton_masks_64[5]
        x = (x ^ (x >>  2)) & morton_masks_64[4]
        x = (x ^ (x >>  4)) & morton_masks_64[3]
        x = (x ^ (x >>  8)) & morton_masks_64[2]
        x = (x ^ (x >> 16)) & morton_masks_64[1]
        return x
    end

    function encode_morton(x::UInt32, y::UInt32)
        x = pdep_0x55555555(x)
        y = pdep_0x55555555(y)
        return x | (y << 1)
    end

    function encode_morton(x::UInt64, y::UInt64)
        x = pdep_0x5555555555555555(x)
        y = pdep_0x5555555555555555(y)
        return x | (y << 1)
    end

    function decode_morton(z::UInt32)
        x = pext_0x55555555(z)
        y = pext_0x55555555(z >> 1)
        return x, y
    end

    function decode_morton(z::UInt64)
        x = pext_0x5555555555555555(z)
        y = pext_0x5555555555555555(z >> 1)
        return x, y
    end
end

# Normalizes the coordinate to the range [0, MORTON_MAX_SIDE] before computing
# the morton index. This allows for a more sane Float encoding, especially for a known
# domain size.
function normalized_encode_morton(x::T, y::T, scale_inv::T) where {T <: AbstractFloat} 
    x_u = floor(MORTON_INDEX_TYPE, x * scale_inv * MORTON_MAX_SIDE )
    y_u = floor(MORTON_INDEX_TYPE, y * scale_inv * MORTON_MAX_SIDE )
    return encode_morton(x_u, y_u)
end

# Decodes a floating point x,y pair, encoded by "normalized_encode_morton", scaling it
# back to the original domain.
function normalized_decode_morton(z::MORTON_INDEX_TYPE, scale::T) where {T <: AbstractFloat} 
    mms_inv = T(1) / MORTON_MAX_SIDE
    x, y = decode_morton(z)
    return x * scale * mms_inv, y * scale * mms_inv
end
