Base.@kwdef mutable struct Tree
    data::Any
    parent::Union{Nothing,Ref{Tree}} = nothing
    children::Vector{Ref{Tree}} = Ref{Tree}[]
    function Tree(data::Any, parent::Union{Nothing,Ref{Tree}}, children::Vector{Ref{Tree}})
        this = new(data, parent, children)
        if parent !== nothing
            push!(parent[].children, Ref(this))
        end
        return this
    end
end
# The level of a node is defined by 1 + the number of connections between the node and the root
function level(tree::Tree; current_level=1)
    if tree.parent !== nothing
        return level(tree.parent[]; current_level = current_level + 1)
    else
        return current_level
    end
end
# Is this the last child in the parent's list of children?
# offset determines if the nth-parent is the last child
function is_last_child(tree::Tree; relative_offset=0)
    if tree.parent == nothing
        return true
    end
    if relative_offset > 0
        return is_last_child(tree.parent[]; relative_offset=relative_offset-1)
    else
        nsiblings = length(tree.parent[].children) - 1
        return (tree.parent[].children[nsiblings + 1][] == tree)
    end
end
function Base.show(io::IO, tree::Tree; relative_offset=0)
    nsiblings = 0
    for i = relative_offset:-1:1
        if i === 1 && is_last_child(tree, relative_offset=i-1)
            print("└─ ")
        elseif i === 1
            print("├─ ")
        elseif is_last_child(tree, relative_offset=i-1)
            print("   ")
        else
            print("│  ")
        end
    end
    println(tree.data)
    for child in tree.children
        show(io, child[]; relative_offset = relative_offset + 1)
    end
end
