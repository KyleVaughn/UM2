export Mat,
       Mat2x2,
       Mat2x2f, Mat2x2d

export col

# Do not declare as AbstractMatrix due to Julia's column-major storage
struct Mat{M, N, T}
    # M rows of size N
    rows::Vec{M, Vec{N, T}}
end

# -- Type aliases --

const Mat2x2  = Mat{2, 2}
const Mat2x2f = Mat{2, 2, Float32}
const Mat2x2d = Mat{2, 2, Float64}

# -- Interface --

Base.getindex(m::Mat, i) = m.rows[i]
Base.getindex(m::Mat, i, j) = m.rows[i][j]
Base.size(m::Mat{M, N}) where {M, N} = (M, N)
Base.length(m::Mat{M, N}) where {M, N} = M * N 
Base.eltype(m::Mat{M, N, T}) where {M, N, T} = T

# -- Constructors --

function Mat{M, N}(xs::T...) where {M, N, T}
    @assert(length(xs) == M * N)
    return Mat{M, N, T}(vec(i->Vec(xs[ (1 + (i - 1) * N) : (i * N) ]), Val(M)))
end

function col(m::Mat{M, N, T}, j) where {M, N, T}
    return vec(i -> m[i, j], Val(M))
end

# -- Unary operators --

Base.:-(m::Mat{M, N, T}) where {M, N, T} = Mat{M, N, T}(vec(i -> -m[i], Val(M)))

# -- Binary operators --

Base.:*(m::Mat{M, N, T}, scalar::X) where {M, N, T, X} = Mat{M, N, T}(vec(i -> m[i] * T(scalar), Val(M)))
Base.:/(m::Mat{M, N, T}, scalar::X) where {M, N, T, X} = Mat{M, N, T}(vec(i -> m[i] / T(scalar), Val(M)))
Base.:*(scalar::X, m::Mat{M, N, T}) where {M, N, T, X} = Mat{M, N, T}(vec(i -> T(scalar) * m[i], Val(M)))
Base.:/(scalar::X, m::Mat{M, N, T}) where {M, N, T, X} = Mat{M, N, T}(vec(i -> T(scalar) / m[i], Val(M)))
Base.:+(lhs::Mat{M, N, T}, rhs::Mat{M, N, T}) where {M, N, T} = Mat{M, N, T}(vec(i -> lhs[i] + rhs[i], Val(M)))
Base.:-(lhs::Mat{M, N, T}, rhs::Mat{M, N, T}) where {M, N, T} = Mat{M, N, T}(vec(i -> lhs[i] - rhs[i], Val(M)))
Base.:*(m::Mat{M, N, T}, v::Vec{N, T}) where {M, N, T} = vec(i -> m[i] ⋅ v, Val(M))
function Base.:*(lhs::Mat{M, N, T}, rhs::Mat{N, M, T}) where {M, N, T}
    return Mat{M, N, T}(
            vec(i -> 
                vec(j -> 
                    lhs[i] ⋅ col(rhs, j), 
                Val(N)), 
            Val(M)))
end

# -- IO --

function Base.show(io::IO, m::Mat{M, N, T}) where {M, N, T}
    println(io, M, '×', N, " Mat{", T, "}")
    for i = 1:M
        for j = 1:N
            print(io,  " ", m.rows[i][j])
        end
        println(io)
    end
end
