export Tree
export isroot, is_parents_last_child

mutable struct Tree{T}
    data::T
    parent::Union{Nothing, Tree{T}}
    children::Union{Nothing, Vector{Tree{T}}}
end

Tree(data::T) where {T} = Tree{T}(data, nothing, nothing)
function Tree(data::T, parent::Tree{T}) where{T}
    this = Tree{T}(data, parent, nothing)
    if isnothing(parent.children)
        parent.children = [this]
    else
        push!(parent.children, this)
    end
    return this
end 

isroot(t::Tree) = t.parent === nothing
is_parents_last_child(t::Tree) = t.parent.children[end] === t

function Base.show(io::IO, tree::Tree)
    @warn "Make sure if there are more than 5 or so children, to just print ..."
    println(io, tree.data)
    if !isnothing(tree.children)
        for child ∈ tree.children
            show(io, child, "") 
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
        for child ∈ tree.children
            show(io, child, next_predecessor_string)
        end
    end
end
