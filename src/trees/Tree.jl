abstract type Tree end

isroot(tree::Tree) = isnothing(tree.parent)

function is_parents_last_child(tree::Tree)
    if isnothing(tree.parent) || tree.parent.children[end] === tree
        return true
    else
        return false
    end
end

leaves(tree::Tree) = leaves!(tree, typeof(tree)[])

function leaves!(node::T, leaf_nodes::Vector{T}) where {T <: Tree}
    if isnothing(node.children)
        push!(leaf_nodes, node)
        return nothing 
    else
        for child in node.children
            leaves!(child, leaf_nodes)
        end
    end
    return leaf_nodes
end

Base.show(io::IO, tree::Tree) = Base.show(io::IO, tree::Tree, "")
function Base.show(io::IO, tree::Tree, predecessor_string::String)
    if isroot(tree)
        next_predecessor_string = ""
    elseif !isroot(tree) || next_predecessor_string != ""
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
