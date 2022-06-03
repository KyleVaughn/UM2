using Jacobi
using Polynomials

# Script for writing the legerdre_RefLine file with Legendre polynomials up to
# a maximum degree of max_degree.
max_degree = 50

setprecision(BigFloat, 512)

function weights(points)
    N = length(points)
    return [2 / ((1 - x^2) * (dlegendre(x, N)^2)) for x in points]
end

open("legendre_line.jl", "w") do io
    for i in 1:max_degree
        # Legendre = Jacobi with α = β = 0
        x = jacobi_zeros(i, 0, 0, BigFloat)
        w = weights(x) / 2
        println("Error in sum of the weights for degree $i: $(1-sum(w))")
        x = (x .+ 1) / 2
        println(io, "@generated function gauss_quadrature(form::Val{:legendre},")
        println(io, "                                     shape::RefLine,")
        println(io, "                                     degree::Val{$i},")
        println(io, "                                     type::Type{T}) where {T}")
        println(io, """    weights = SVector(:(\$(T(big"$(w[1])"))),""")
        for j in 2:i
            println(io, """                      :(\$(T(big"$(w[j])"))),""")
        end
        println(io, "                       )")
        println(io, """    points = SVector(:(\$(NTuple{1,T}(big"$(x[1])"))),""")
        for j in 2:i
            println(io, """                     :(\$(NTuple{1,T}(big"$(x[j])"))),""")
        end
        println(io, "                      )")
        println(io, "    return weights, points")
        println(io, "end")
    end
end
