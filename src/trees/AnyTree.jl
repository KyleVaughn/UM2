"""
    AnyTree(data)
    AnyTree(data, parent)

A tree data structure to hold arbitrary data, parent, and children.
"""
mutable struct AnyTree <: Tree
    data::Union{Nothing, Any}
    parent::Union{Nothing, Any}
    children::Union{Nothing, Vector{Any}}
    AnyTree(data) = new(data, nothing, nothing)
    function AnyTree(data, parent::AnyTree)
        this = new(data, parent, nothing)
        if isnothing(parent.children)
            parent.children = [this]
        else
            push!(parent.children, this)
        end 
        return this
    end
end
