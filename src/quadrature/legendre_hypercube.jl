for quad_degree in 1:20 # Exceeding degree 20 seems unnecessary at this time
    @eval begin
        # Square
        @generated function gauss_quadrature(form::LegendreType,
                                             shape::RefSquare,
                                             degree::Val{$quad_degree},
                                             type::Type{T}) where {T}
            # Is there a way to use form and degree here instead?
            line_weights, line_points = gauss_quadrature(LegendreType(),
                                                         RefLine(),
                                                         Val($quad_degree),
                                                         T)
            weights = Expr[]
            points = Expr[]
            for i in 1:($quad_degree)
                for j in 1:($quad_degree)
                    push!(weights, :($(line_weights[i]) * $(line_weights[j])))
                    push!(points, :(tuple($(line_points[i][1]), $(line_points[j][1]))))
                end
            end
            return quote
                return (SVector{$(length(line_weights)^2), $T}(tuple($(weights...))),
                        SVector{$(length(line_weights)^2), NTuple{2, $T}}(tuple($(points...))))
            end
        end
        # Cube
        @generated function gauss_quadrature(form::LegendreType,
                                             shape::RefCube,
                                             degree::Val{$quad_degree},
                                             type::Type{T}) where {T}
            # Is there a way to use form and degree here instead?
            line_weights, line_points = gauss_quadrature(LegendreType(),
                                                         RefLine(),
                                                         Val($quad_degree),
                                                         T)
            weights = Expr[]
            points = Expr[]
            for i in 1:($quad_degree)
                for j in 1:($quad_degree)
                    for k in 1:($quad_degree)
                        push!(weights,
                              :($(line_weights[i]) *
                                $(line_weights[j]) *
                                $(line_weights[k])))
                        push!(points,
                              :(tuple($(line_points[i][1]),
                                      $(line_points[j][1]),
                                      $(line_points[k][1]))))
                    end
                end
            end
            return quote
                return (SVector{$(length(line_weights)^3), $T}(tuple($(weights...))),
                        SVector{$(length(line_weights)^3), NTuple{3, $T}}(tuple($(points...))))
            end
        end
    end
end
