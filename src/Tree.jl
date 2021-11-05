mutable struct Tree
    data::Any
    parent::Ref{Tree}
    children::Vector{Ref{Tree}}
end

function Tree(;data::Any = nothing, 
               parent::Ref{Tree} = Ref{Tree}(), 
               children::Vector{Ref{Tree}} = Ref{Tree}[])
    this = Tree(data, parent, children)
    if isassigned(parent)
        push!(parent[].children, Ref(this))
    end
    return this
end

# The level of a node is defined by 1 + the number of connections between the node and the root
function get_level(tree::Tree; current_level=1)
    if isassigned(tree.parent)
        return get_level(tree.parent[]; current_level = current_level + 1)
    else
        return current_level
    end
end
# Is this the last child in the parent's list of children?
# offset determines if the nth-parent is the last child
function _is_last_child(tree::Tree; relative_offset=0)
    if !iassigned(tree.parent)
        return true
    end
    if relative_offset > 0
        return _is_last_child(tree.parent[]; relative_offset=relative_offset-1)
    else
        nsiblings = length(tree.parent[].children) - 1
        return (tree.parent[].children[nsiblings + 1][] == tree)
    end
end
function Base.show(io::IO, tree::Tree; relative_offset=0)
    nsiblings = 0
    for i = relative_offset:-1:1
        if i === 1 && _is_last_child(tree, relative_offset=i-1)
            print(io, "└─ ")
        elseif i === 1
            print(io, "├─ ")
        elseif _is_last_child(tree, relative_offset=i-1)
            print(io, "   ")
        else
            print(io, "│  ")
        end
    end
    println(io, tree.data)
    for child in tree.children
        show(io, child[]; relative_offset = relative_offset + 1)
    end
end
