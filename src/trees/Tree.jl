mutable struct Tree
    data::Union{Nothing, Any}
    parent::Union{Nothing, Tree}
    children::Union{Nothing, Vector{Tree}}
    Tree(data) = new(data, nothing, nothing)
    function Tree(data, parent::Tree)
        this = new(data, parent, nothing)
        if isnothing(parent.children)
            parent.children = [this]
        else
            push!(parent.children, this)
        end 
        return this
    end
end

is_root(tree::Tree) = isnothing(tree.parent)

function is_parents_last_child(tree::Tree)
    if isnothing(tree.parent) || tree.parent.children[end] === tree
        return true
    else
        return false
    end
end

Base.show(io::IO, tree::Tree) = Base.show(io::IO, tree::Tree, "")
function Base.show(io::IO, tree::Tree, predecessor_string::String)
    if is_root(tree)
        next_predecessor_string = ""
    elseif !is_root(tree) || next_predecessor_string != ""
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
