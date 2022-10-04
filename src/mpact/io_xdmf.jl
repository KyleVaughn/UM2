export write_xdmf

#################################################################################
#                                    WRITE
#################################################################################

function write_xdmf(mmh::MPACTMeshHierarchy{M},
                    elsets::Dict{String, Set{I}},
                    filename::String) where {M, I}
    # Check valid filename
    if !endswith(filename, ".xdmf")
        error("Invalid filename.")
    end
    # h5 filename
    h5_filename = filename[1:(end - 4)] * "h5"
    h5_file = h5open(h5_filename, "w")
    # XML
    xdoc = XMLDocument()
    try
        # Xdmf
        xroot = ElementNode("Xdmf")
        setroot!(xdoc, xroot)
        link!(xroot, AttributeNode("Version", "3.0"))
        # Domain
        xdomain = ElementNode("Domain")
        link!(xroot, xdomain)
        # Material names
        # All leaf meshes should have same material_names
        material_names = String[]    
        for elset_name in keys(elsets)    
            if startswith(elset_name, "Material:")    
                push!(material_names, elset_name[11:end])    
            end    
        end                                            
        sort!(material_names) 
        if 0 < length(material_names)
            xmaterials = ElementNode("Information")
            link!(xdomain, xmaterials)
            link!(xmaterials, AttributeNode("Name", "Materials"))
            link!(xmaterials, TextNode(join(mpt.leaf_meshes[1].material_names, " ")))
        end
        _add_mesh_partition_xdmf!(xdomain, h5_filename, h5_file,
                                  mpt.partition_tree, mpt.leaf_meshes)
    finally
        write(filename, xdoc)
        close(h5_file)
    end
    return nothing
end

#function _add_mesh_partition_xdmf!(xml::EzXML.Node,
#                                   h5_filename::String,
#                                   h5_mesh::Union{HDF5.Group, HDF5.File},
#                                   node::Tree{Tuple{Int64, String}},
#                                   leaf_meshes::Vector{<:AbstractMesh})
#    if !isleaf(node) # Internal node
#        name = node.data[2]
#        # Grid
#        xgrid = ElementNode("Grid")
#        link!(xml, xgrid)
#        link!(xgrid, AttributeNode("Name", name))
#        link!(xgrid, AttributeNode("GridType", "Tree"))
#        # h5_group
#        h5_group = create_group(h5_mesh, name)
#        for child in node.children
#            _add_mesh_partition_xdmf!(xgrid, h5_filename, h5_group, child, leaf_meshes)
#        end
#    else # Leaf node
#        id = node.data[1]
#        _add_uniform_grid_xdmf!(xml, h5_filename, h5_mesh, leaf_meshes[id])
#    end
#    return nothing
#end

##################################################################################
##                                    READ
##################################################################################
#xdmf_read_error(x::String) = error("Error reading XDMF file.")
#function read_xdmf(path::String, ::Type{T}) where {T <: AbstractFloat}
#    xdoc = readxml(path)
#    xroot = root(xdoc)
#    nodename(xroot) != "Xdmf" && xdmf_read_error()
#    h5_file = h5open(path[begin:(end - 4)] * "h5", "r")
#    try
#        version = xroot["Version"]
#        version != "3.0" && xdmf_read_error()
#        xdomain = firstnode(xroot)
#        nodename(xdomain) != "Domain" && xdmf_read_error()
#        material_names = String[]
#        nnodes = countnodes(xdomain)
#        if 2 == nnodes && nodename(firstnode(xdomain)) == "Information"
#            append!(material_names, split(nodecontent(firstnode(xdomain)), " "))
#            xgrid = nodes(xdomain)[2]
#        elseif 1 == nnodes
#            xgrid = firstnode(xdomain)
#        else
#            xdmf_read_error()
#        end
#        grid_type = xgrid["GridType"]
#        if grid_type == "Uniform"
#            return _read_xdmf_uniform_grid(xgrid, h5_file, material_names)
#        elseif grid_type == "Tree"
#            # Create tree
#            root = Tree((1, xgrid["Name"]))
#            _setup_xdmf_tree!(xgrid, root, [0])
#            nleaf_meshes = nleaves(root)
#            dim, float_type, uint_type = _get_volume_mesh_params_from_xdmf(xgrid, h5_file)
#            leaf_meshes = Vector{VolumeMesh{dim, float_type, uint_type}}(undef,
#                                                                         nleaf_meshes)
#            # fill the leaf meshes
#            nleaf = _setup_xdmf_leaf_meshes!(xgrid, h5_file, 1, leaf_meshes, material_names)
#            @assert nleaf - 1 == nleaf_meshes
#            return MeshPartitionTree(root, leaf_meshes)
#        else
#            xdmf_read_error()
#        end
#    finally
#        close(h5_file)
#    end
#    return nothing
#end
#
## Helper functions for read_xdmf
## -------------------------------------------------------------------------------------------------
#function _read_xdmf_uniform_grid(xgrid::EzXML.Node,
#                                 h5_file::HDF5.File,
#                                 material_names::Vector{String})
#    # Get all the h5 file paths to relevant data
#    points_path = ""
#    connectivity_path = ""
#    material_path = ""
#    group_paths = String[]
#    group_names = String[]
#    for child in eachnode(xgrid)
#        child_name = nodename(child)
#        m = match(r"(?<=:/).*", nodecontent(child))
#        path = string(m.match)
#        if child_name == "Geometry"
#            points_path = path
#        elseif child_name == "Topology"
#            connectivity_path = string(m.match)
#        elseif child_name == "Attribute" && haskey(child, "Name") &&
#               child["Name"] == "Material"
#            material_path = path
#        elseif child_name == "Set" && haskey(child, "SetType") && child["SetType"] == "Cell"
#            push!(group_paths, path)
#            push!(group_names, child["Name"])
#        else
#            warn("Unused XML node: " * child_name)
#        end
#    end
#    # Points
#    name = xgrid["Name"]
#    points_xyz = read(h5_file[points_path])
#    dim, npoints = size(points_xyz)
#    float_type = eltype(points_xyz)
#    points = collect(reinterpret(reshape, Point{dim, float_type}, points_xyz))
#    # Connectivity
#    connectivity = read(h5_file[connectivity_path])
#    uint_type = eltype(connectivity)
#    # count elements
#    conn_length = length(connectivity)
#    nelements = 0
#    offset = 1
#    while offset < conn_length
#        nelements += 1
#        xdmf_type = connectivity[offset]
#        vtk_type = xdmf2vtk(xdmf_type)
#        offset += points_in_vtk_type(vtk_type) + 1
#    end
#    # set offsets and types
#    offsets = Vector{uint_type}(undef, nelements + 1)
#    offset = 1
#    for i in 1:nelements
#        offsets[i] = offset
#        xdmf_type = connectivity[offset]
#        vtk_type = xdmf2vtk(xdmf_type)
#        offset += points_in_vtk_type(vtk_type) + 1
#    end
#    deleteat!(connectivity, view(offsets, 1:nelements))
#    connectivity .+= 1 # convert 0-based to 1-based
#    # Account for deletion
#    for i in 1:nelements
#        offsets[i] -= i - 1
#    end
#    # Add the final offset
#    offsets[nelements + 1] = length(connectivity) + 1
#    # Materials
#    materials = zeros(UInt8, nelements)
#    if material_path != ""
#        materials[:] = read(h5_file[material_path]) .+= 1
#    end
#    # Groups
#    groups = Dict{String, BitSet}()
#    for i in 1:length(group_paths)
#        groups[group_names[i]] = BitSet(read(h5_file[group_paths[i]]) .+= 1)
#    end
#    return VolumeMesh{dim, float_type, uint_type}(points, offsets, connectivity,
#                                                  materials, material_names, name, groups)
#end
#
#function _setup_xdmf_tree!(xmlnode::EzXML.Node,
#                          treenode::Tree{Tuple{Int64, String}},
#                          ids::Vector{Int64},
#                          level::Int64 = 0)
#    level += 1
#    if length(ids) < level
#        push!(ids, 0)
#    end
#    if hasnode(xmlnode)
#        for xmlchild in eachnode(xmlnode)
#            if nodename(xmlchild) == "Grid"
#                ids[level] += 1
#                treechild = Tree((ids[level], xmlchild["Name"]), treenode)
#                _setup_xdmf_tree!(xmlchild, treechild, ids, level)
#            end
#        end
#    end
#    return nothing
#end
#
#function _get_volume_mesh_params_from_xdmf(xgrid::EzXML.Node,
#                                           h5_file::HDF5.File)
#    xmlchild = firstnode(xgrid)
#    max_iters = 5
#    i = 1
#    while i ≤ max_iters
#        if nodename(xmlchild) == "Grid" && xmlchild["GridType"] == "Uniform"
#            points_path = ""
#            connectivity_path = ""
#            for child in eachnode(xmlchild)
#                child_name = nodename(child)
#                m = match(r"(?<=:/).*", nodecontent(child))
#                path = string(m.match)
#                if child_name == "Geometry"
#                    points_path = path
#                elseif child_name == "Topology"
#                    connectivity_path = string(m.match)
#                end
#                if points_path != "" && connectivity_path != ""
#                    break
#                end
#            end
#            # Points
#            points_xyz = read(h5_file[points_path])
#            dim, npoints = size(points_xyz)
#            float_type = eltype(points_xyz)
#            # Connectivity
#            connectivity = read(h5_file[connectivity_path])
#            uint_type = eltype(connectivity)
#            return (dim, float_type, uint_type)
#        end
#        xmlchild = firstnode(xmlchild)
#        i += 1
#    end
#    return error("Could not determine volume mesh parameters.")
#end
#
#function _setup_xdmf_leaf_meshes!(xmlnode::EzXML.Node,
#                                  h5_file::HDF5.File,
#                                  idx::Int64,
#                                  leaf_meshes::Vector{<:VolumeMesh},
#                                  material_names::Vector{String})
#    id = idx
#    if hasnode(xmlnode)
#        for xmlchild in eachnode(xmlnode)
#            if nodename(xmlchild) == "Grid"
#                if xmlchild["GridType"] == "Uniform"
#                    leaf_meshes[id] = _read_xdmf_uniform_grid(xmlchild, h5_file,
#                                                              material_names)
#                    id += 1
#                elseif xmlchild["GridType"] == "Tree"
#                    id = _setup_xdmf_leaf_meshes!(xmlchild, h5_file, id,
#                                                  leaf_meshes, material_names)
#                else
#                    error("Unsupported GridType")
#                end
#            end
#        end
#    end
#    return id
#end
