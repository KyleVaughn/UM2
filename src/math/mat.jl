export Mat,
       Mat2x2,
       Mat2x2f, Mat2x2d

export col

# MATRIX    
# ---------------------------------------------------------------------------    
#    
# An M x N matrix with data of type T    
#  

struct Mat{M, N, T}
    cols::Vec{N, Vec{M, T}}
end

# -- Type aliases --

const Mat2x2  = Mat{2, 2}
const Mat2x2f = Mat{2, 2, Float32}
const Mat2x2d = Mat{2, 2, Float64}

# -- Interface --

Base.getindex(m::Mat, i) = m.cols[i]
Base.getindex(m::Mat, i, j) = m.cols[j][i]
Base.size(m::Mat{M, N}) where {M, N} = (M, N)
Base.length(m::Mat{M, N}) where {M, N} = M * N 
Base.eltype(m::Mat{M, N, T}) where {M, N, T} = T

# -- Constructors --

function Mat{M, N}(xs::T...) where {M, N, T}
    @assert(length(xs) == M * N)
    return Mat{M, N, T}(vec(i->Vec(xs[ (1 + (i - 1) * M) : (i * M) ]), Val(N)))
end

function Mat(vecs::Vec...)
    return Mat(Vec(vecs))
end

# -- Unary operators --

Base.:-(m::Mat{M, N, T}) where {M, N, T} = Mat{M, N, T}(vec(i -> -m[i], Val(M)))

# -- Binary operators --

function Base.:*(m::Mat{M, N, T}, scalar::X) where {M, N, T, X}
    return Mat{M, N, T}(vec(i -> m[i] * T(scalar), Val(M)))
end

function Base.:/(m::Mat{M, N, T}, scalar::X) where {M, N, T, X}
    return Mat{M, N, T}(vec(i -> m[i] / T(scalar), Val(M)))
end
    
function Base.:*(scalar::X, m::Mat{M, N, T}) where {M, N, T, X}
    return Mat{M, N, T}(vec(i -> T(scalar) * m[i], Val(M)))
end

function Base.:/(scalar::X, m::Mat{M, N, T}) where {M, N, T, X}
    return Mat{M, N, T}(vec(i -> T(scalar) / m[i], Val(M)))
end

function Base.:+(lhs::Mat{M, N, T}, rhs::Mat{M, N, T}) where {M, N, T}
    return Mat{M, N, T}(vec(i -> lhs[i] + rhs[i], Val(M)))
end

function Base.:-(lhs::Mat{M, N, T}, rhs::Mat{M, N, T}) where {M, N, T}
    return Mat{M, N, T}(vec(i -> lhs[i] - rhs[i], Val(M)))
end

# -- 2x2 matrix --


# Provide more opportunity for optimization by explicitly writing out the
# matrix-vector multiplication.
function Base.:*(m::Mat2x2, v::Vec2)
    return Vec2(m[1,1] * v[1] + m[1,2] * v[2],
                m[2,1] * v[1] + m[2,2] * v[2])
end

# -- IO --

function Base.show(io::IO, m::Mat{M, N, T}) where {M, N, T}
    println(io, M, '×', N, " Mat{", T, "}")
    for i = 1:M
        for j = 1:N
            print(io,  " ", m[i, j])
        end
        println(io)
    end
end
