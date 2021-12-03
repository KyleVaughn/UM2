const L_system_instructions = @SVector ["F", # Go forward
                                        "-", # Turn right 90°
                                        "+"  # Turn left 90°
                                       ]

struct L_System
    axiom::String
    rules::Dict{String, String}
end

function apply_rules(L::L_System, N::Int64; production_sequence::String = "")
    if production_sequence == ""
        production_sequence = L.axiom
    end
    if N == 0
        return production_sequence
    else
        new_production_sequence = String[] 
        for i in production_sequence
            if i ∈  L_system_instructions
                push!(new_production_sequence, string(i))
            else
                push!(new_production_sequence, L.rules[string(i)])
            end
        end
        return apply_rules(L, N-1, production_sequence = join(new_production_sequence))
    end
end
