const L_system_instructions = @SVector ['F', # Go forward
                                        '-', # Turn right 90°
                                        '+'  # Turn left 90°
                                       ]
struct L_System
    axiom::Char
    rules::Dict{Char, String}
end

function apply_rules(L::L_System, N::Int64; production_sequence::String = "")
    if production_sequence == ""
        production_sequence = string(L.axiom)
    end
    if N == 0
        return production_sequence
    else
        new_production_sequence = String[] 
        for i ∈  production_sequence
            if i ∈  L_system_instructions
                push!(new_production_sequence, string(i))
            else
                push!(new_production_sequence, L.rules[i])
            end
        end
        return apply_rules(L, N-1, production_sequence = join(new_production_sequence))
    end
end

# For rectangular domains, with angles constrained to n(π/2) where n = 0,1,2,...
# θ₀ = 0 ⟹     0°
# θ₀ = 1 ⟹    90°
# θ₀ = 2 ⟹   180°
# θ₀ = 3 ⟹   270°
function generate_points(production_sequence::String, p₀::Point_2D, 
                         θ₀::Int64, Δx::Float64, Δy::Float64)
    points = [p₀]
    p = p₀
    θ = θ₀
    for i ∈  production_sequence
        if i == 'F'
            p = p + Point_2D(cos(θ*π/2)*Δx, sin(θ*π/2)*Δy)
            push!(points, p)
        elseif i == '+'
            θ = mod(θ + 1, 4)
        elseif i == '-'
            θ = mod(θ - 1, 4)
        end
    end
    return points
end

function hilbert_curve()
    # Note Hilbert curve returns 2^(n^2-1) points
    return L_System('A', Dict('A' => "+BF-AFA-FB+", 'B' => "-AF+BFB+FA-"))
end

# Generate the vector of points making up the Hilbert curve that fill a bounding box bb
# npoints is the target number of points, where the number of points returned will always
# be greater than or equal to npoints
function hilbert_curve(bb::Rectangle_2D, npoints::Int64)
    L = hilbert_curve() 
    # Note Hilbert curve returns 2^(2n) points
    n = ceil(Int64, log(npoints)/(2log(2)))
    npoints_actual = 2^(2n)
    Δx = width(bb)/(sqrt(npoints_actual) - 1)
    Δy = height(bb)/(sqrt(npoints_actual) - 1)
    production_sequence = apply_rules(L, n)
    return generate_points(production_sequence, bb.bl, 0, Δx, Δy) 
end
