mutable struct Tree{T}
    data::Union{Nothing, T}
    parent::Union{Nothing, Tree{T}}
    children::Union{Nothing, Vector{Tree{T}}}
    Tree(data::T) where T = new{T}(data, nothing, nothing)
    function Tree(data::T, parent::Tree{T}) where T
        this = new{T}(data, parent, nothing)
        if isnothing(parent.children)
            parent.children = [this]
        else
            push!(parent.children, this)
        end 
        return this
    end
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
