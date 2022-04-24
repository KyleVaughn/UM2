for quad_degree = 1:30 # current max degree on RefLine
    @eval begin
        # Square
        @generated function gauss_quadrature(form::Val{:legendre},
                                  shape::RefSquare,
                                  degree::Val{$quad_degree},
                                  type::Type{T}) where {T}
            # Is there a way to use form and degree here instead?
            line_weights, line_points = gauss_quadrature(Val(:legendre), 
                                                         RefLine(),
                                                         Val($quad_degree),
                                                         T)
            weights = Expr[] 
            points = Expr[] 
            for i = 1:$quad_degree
                for j = 1:$quad_degree
                    push!(weights, :($(line_weights[i]) * $(line_weights[j])))
                    push!(points, :(Point($(line_points[i][1]), $(line_points[j][1]))))
                end
            end
            return quote
                return (Vec{$(length(line_weights)^2), $T}(tuple($(weights...))),
                        Vec{$(length(line_weights)^2), Point{2,$T}}(tuple($(points...))))
            end
        end
        # Cube
        @generated function gauss_quadrature(form::Val{:legendre},
                                  shape::RefCube,
                                  degree::Val{$quad_degree},
                                  type::Type{T}) where {T}
            # Is there a way to use form and degree here instead?
            line_weights, line_points = gauss_quadrature(Val(:legendre), 
                                                         RefLine(),
                                                         Val($quad_degree),
                                                         T)
            weights = Expr[] 
            points = Expr[] 
            for i = 1:$quad_degree
                for j = 1:$quad_degree
                    for k = 1:$quad_degree
                        push!(weights, :($(line_weights[i]) * 
                                         $(line_weights[j]) *
                                         $(line_weights[k])
                                        ))
                        push!(points, :(Point($(line_points[i][1]), 
                                              $(line_points[j][1]),
                                              $(line_points[k][1]),
                                             )))
                    end
                end
            end
            return quote
                return (Vec{$(length(line_weights)^3), $T}(tuple($(weights...))),
                        Vec{$(length(line_weights)^3), Point{3,$T}}(tuple($(points...))))
            end
        end
    end
end
