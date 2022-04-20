for dim ∈ 2:3
    for qorder = 1:25 # current max order on RefLine
        @eval begin
            function gauss_quadrature(form::Val{:legendre},
                                      shape::RefHypercube{$dim},
                                      order::Val{$qorder},
                                      type::Type{T}) where {T}
                line_weights, line_points = gauss_quadrature(form, RefLine(),order,T)
                weights = kron(weights, weights)
                points = map
                return line_weights, line_points
            end
        end
    end
end
