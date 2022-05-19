export Tree
export isroot, is_parents_last_child, leaves, parent, children

mutable struct Tree{T}
    data::T
    parent::Union{Nothing, Tree{T}}
    children::Union{Nothing, Vector{Tree{T}}}
end

parent(t::Tree) = t.parent
children(t::Tree) = t.children

Tree(data::T) where {T} = Tree{T}(data, nothing, nothing)
function Tree(data::T, parent::Tree{T}) where{T}
    this = Tree{T}(data, parent, nothing)
    if isnothing(children(parent))
        parent.children = [this]
    else
        push!(parent.children, this)
    end
    return this
end 

isroot(t::Tree) = parent(t) === nothing
is_parents_last_child(t::Tree) = children(parent(t))[end] === t
function leaves(tree::Tree{T}) where {T}
    leaf_nodes = Tree{T}[]
    if !isnothing(children(tree))
        for child ∈ tree.children
            get_leaves!(child, leaf_nodes)
        end
    else
        push!(leaf_nodes, tree)
    end
    return leaf_nodes
end
function get_leaves!(tree::Tree{T}, leaf_nodes::Vector{Tree{T}}) where {T}
    if !isnothing(tree.children)
        for child ∈ tree.children
            get_leaves!(child, leaf_nodes)
        end
    else
        push!(leaf_nodes, tree)
    end
    return nothing
end

function Base.show(io::IO, tree::Tree)
    println(io, tree.data)
    if !isnothing(tree.children)
        nchildren = length(tree.children)
        if 7 < nchildren 
            show(io, tree.children[1], "")
            show(io, tree.children[2], "")
            println(io, "│  ⋮") 
            println(io, "│  ⋮ (", nchildren-4, " additional children)")
            println(io, "│  ⋮")
            show(io, tree.children[end-1], "") 
            show(io, tree.children[end],   "") 
        else
            for child ∈ tree.children
                show(io, child, "") 
            end
        end
    end
end
function Base.show(io::IO, tree::Tree, predecessor_string::String)
    next_predecessor_string = ""
    if !isroot(tree)
        last_child = is_parents_last_child(tree)
        if is_parents_last_child(tree)
            print(io, predecessor_string * "└─ ")
            next_predecessor_string = predecessor_string * "   "
        else
            print(io, predecessor_string * "├─ ")
            next_predecessor_string = predecessor_string * "│  "
        end
    end
    println(io, tree.data)
    if !isnothing(tree.children)
        nchildren = length(tree.children)
        if 7 < nchildren 
            show(io, tree.children[1], next_predecessor_string)
            show(io, tree.children[2], next_predecessor_string)
            println(io, next_predecessor_string * "│  ⋮") 
            println(io, next_predecessor_string * "│  ⋮ (", nchildren-4, 
                    " additional children)")
            println(io, next_predecessor_string * "│  ⋮")
            show(io, tree.children[end-1], next_predecessor_string)
            show(io, tree.children[end], next_predecessor_string)
        else
            for child ∈ tree.children
                show(io, child, next_predecessor_string)
            end
        end
    end
end
