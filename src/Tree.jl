mutable struct Tree{T}
    data::Union{Nothing, T}
    parent::Union{Nothing, Tree{T}}
    children::Union{Nothing, Vector{Tree{T}}}
end

function Tree{T}(;data::Union{Nothing, T} = nothing, 
                 parent::Union{Nothing, Tree{T}} = nothing, 
                 children::Union{Nothing, Vector{Tree{T}}} = nothing) where T
    this = Tree{T}(data, parent, children)
    if !isnothing(parent)
        if isnothing(parent.children)
            parent.children = [this]
        else
            push!(parent.children, this)
        end
    end
    return this
end

function is_parents_last_child(tree::Tree)
    if isnothing(tree.parent) || tree.parent.children[end] === tree
        return true
    else
        return false
    end
end

Base.show(io::IO, tree::Tree) = Base.show(io::IO, tree::Tree, "")
function Base.show(io::IO, tree::Tree, predecessor_string::String)
    if predecessor_string !== ""
        last_child = is_parents_last_child(tree)
        if last_child
            print(io, predecessor_string * "└─ ")
            next_predecessor_string = predecessor_string * "   "
        else
            print(io, predecessor_string * "├─ ")
            next_predecessor_string = predecessor_string * "│  "
        end
    else
        next_predecessor_string = "   "
    end
    println(io, tree.data)
    if !isnothing(tree.children)
        for child in tree.children
            show(io, child, next_predecessor_string)
        end
    end
end
