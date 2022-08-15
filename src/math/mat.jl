export Mat,
       Mat2x2,
       Mat2x2f, Mat2x2d

export det

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

Base.getindex(A::Mat, i) = A.cols[i]
Base.getindex(A::Mat, i, j) = A.cols[j][i]
Base.size(A::Mat{M, N}) where {M, N} = (M, N)
Base.length(A::Mat{M, N}) where {M, N} = M * N 
Base.eltype(A::Mat{M, N, T}) where {M, N, T} = T

# -- Constructors --

# Seems to allocate sometimes. Commenting out for now.
#function Mat{M, N}(xs::T...) where {M, N, T}
#    @assert(length(xs) == M * N)
#    return Mat{M, N, T}(vec(i->Vec{M, T}(xs[ (1 + (i - 1) * M) : (i * M) ]), Val(N)))
#end

function Mat(vecs::Vec...)
    return Mat(Vec(vecs))
end

# -- Unary operators --

Base.:-(A::Mat{M, N, T}) where {M, N, T} = Mat{M, N, T}(vec(i -> -A[i], Val(M)))

# -- Binary operators --

function Base.:*(A::Mat{M, N, T}, scalar::X) where {M, N, T, X}
    return Mat{M, N, T}(vec(i -> A[i] * T(scalar), Val(M)))
end

function Base.:/(A::Mat{M, N, T}, scalar::X) where {M, N, T, X}
    scalar_inv = 1 / T(scalar)
    return Mat{M, N, T}(vec(i -> scalar_inv * A[i], Val(M)))
end
    
function Base.:*(scalar::X, A::Mat{M, N, T}) where {M, N, T, X}
    return Mat{M, N, T}(vec(i -> T(scalar) * A[i], Val(M)))
end

function Base.:/(scalar::X, A::Mat{M, N, T}) where {M, N, T, X}
    return Mat{M, N, T}(vec(i -> T(scalar) / A[i], Val(M)))
end

function Base.:+(A::Mat{M, N, T}, B::Mat{M, N, T}) where {M, N, T}
    return Mat{M, N, T}(vec(i -> A[i] + B[i], Val(M)))
end

function Base.:-(A::Mat{M, N, T}, B::Mat{M, N, T}) where {M, N, T}
    return Mat{M, N, T}(vec(i -> A[i] - B[i], Val(M)))
end

# -- 2x2 matrix --

function Mat2x2(a11::T, a21::T, a12::T, a22::T) where {T}
    return Mat(Vec(a11, a21), Vec(a12, a22))
end

function Base.:*(A::Mat2x2, x::Vec2)
    return Vec2(A[1,1] * x[1] + A[1,2] * x[2],
                A[2,1] * x[1] + A[2,2] * x[2])
end

function Base.:*(A::Mat2x2, B::Mat2x2)
    return Mat2x2(A[1,1] * B[1,1] + A[1,2] * B[2,1],
                  A[2,1] * B[1,1] + A[2,2] * B[2,1],
                  A[1,1] * B[1,2] + A[1,2] * B[2,2],
                  A[2,1] * B[1,2] + A[2,2] * B[2,2])
end

function det(A::Mat2x2)
    return A[1,1] * A[2,2] - A[2,1] * A[1,2] 
end

function Base.inv(A::Mat2x2)
    detinv = 1 / det(A)
    return Mat2x2( detinv * A[2,2], 
                  -detinv * A[2,1],
                  -detinv * A[1,2],  
                   detinv * A[1,1])
end

# -- IO --

function Base.show(io::IO, A::Mat{M, N, T}) where {M, N, T}
    println(io, M, '×', N, " Mat{", T, "}")
    for i = 1:M
        for j = 1:N
            print(io,  " ", A[i, j])
        end
        println(io)
    end
end
