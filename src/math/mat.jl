export Mat,
       Mat2x2,
       Mat2x2f, Mat2x2d

export det

# MATRIX    
# ---------------------------------------------------------------------------    
#    
# An M x N matrix with data of type T    
#  

struct Mat{M, N, T <: AbstractFloat}
    cols::NTuple{N, Vec{M, T}}
end

# -- Type aliases --

const Mat2x2  = Mat{2, 2}
const Mat2x2f = Mat{2, 2, f32}
const Mat2x2d = Mat{2, 2, f64}

# -- Interface --

Base.getindex(A::Mat, i::Integer) = A.cols[i]
Base.getindex(A::Mat, i::Integer, j::Integer) = A.cols[j][i]
Base.size(A::Mat{M, N}) where {M, N} = (M, N)
Base.length(A::Mat{M, N}) where {M, N} = M * N 
Base.eltype(A::Mat{M, N, T}) where {M, N, T} = T

# -- Constructors --

function Mat(vecs::Vec...)
    return Mat(vecs)
end

# -- Unary operators --

Base.:-(A::Mat) = Mat(map(-, A.cols))

# -- Binary operators --

Base.:*(scalar::X, A::Mat{M, N, T}) where {M, N, T, X <: Number} = Mat(map(x -> T(scalar) * x, A.cols))
Base.:*(A::Mat{M, N, T}, scalar::X) where {M, N, T, X <: Number} = scalar * A
Base.:/(A::Mat{M, N, T}, scalar::X) where {M, N, T, X <: Number} = Mat(map(x -> x / T(scalar), A.cols))
Base.:+(A::Mat{M, N, T}, B::Mat{M, N, T}) where {M, N, T} = Mat(map(+, A.cols, B.cols))
Base.:-(A::Mat{M, N, T}, B::Mat{M, N, T}) where {M, N, T} = Mat(map(-, A.cols, B.cols))

# -- 2x2 matrix --

function Mat2x2(a11::T, a21::T, a12::T, a22::T) where {T}
    return Mat(Vec(a11, a21), Vec(a12, a22))
end

function Base.:*(A::Mat2x2, x::Vec2)
    return Vec2(A[1, 1] * x[1] + A[1, 2] * x[2],
                A[2, 1] * x[1] + A[2, 2] * x[2])
end

function Base.:*(A::Mat2x2, B::Mat2x2)
    return Mat2x2(A[1, 1] * B[1, 1] + A[1, 2] * B[2, 1],
                  A[2, 1] * B[1, 1] + A[2, 2] * B[2, 1],
                  A[1, 1] * B[1, 2] + A[1, 2] * B[2, 2],
                  A[2, 1] * B[1, 2] + A[2, 2] * B[2, 2])
end

function det(A::Mat2x2)
    return A[1, 1] * A[2, 2] - A[2, 1] * A[1, 2] 
end

function Base.inv(A::Mat2x2)
    return Mat2x2( A[2, 2], 
                  -A[2, 1],
                  -A[1, 2],  
                   A[1, 1]) / det(A)
end

# -- IO --

function Base.show(io::IO, A::Mat{M, N, T}) where {M, N, T}
    println(io, M, '×', N, " Mat{", T, "}:")
    for i = 1:M
        for j = 1:N
            print(io,  " ", A[i, j])
        end
        println(io)
    end
end
