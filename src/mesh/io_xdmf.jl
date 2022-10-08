export write_xdmf_file!, read_xdmf_file

#################################################################################
#                                    WRITE
#################################################################################

# Helper functions for write_xdmf
# -------------------------------------------------------------------------------------------------
function _add_uniform_grid_xdmf!(xml::EzXML.Node,
                                 h5_filepath::String,
                                 h5_mesh::Union{HDF5.Group, HDF5.File},
                                 mesh::MeshFile{T, I}) where {T <: AbstractFloat, I <: Integer}
    # Grid
    xgrid = ElementNode("Grid")
    link!(xml, xgrid)
    link!(xgrid, AttributeNode("Name", mesh.name))
    link!(xgrid, AttributeNode("GridType", "Uniform"))
    # h5
    h5_group = create_group(h5_mesh, mesh.name)
    # Geometry
    _write_xdmf_geometry!(xgrid, h5_filepath, h5_group, mesh)
    # Topology
    _write_xdmf_topology!(xgrid, h5_filepath, h5_group, mesh)
    # Non-material groups
    _write_xdmf_groups!(xgrid, h5_filepath, h5_group, mesh)
    # Materials
    _write_xdmf_materials!(xgrid, h5_filepath, h5_group, mesh)
    return nothing
end

function _write_xdmf_geometry!(xml::EzXML.Node,
                               h5_filepath::String,
                               h5_mesh::HDF5.Group,
                               mesh::MeshFile{T, I}) where {T <: AbstractFloat, I <: Integer}
    verts = mesh.nodes
    float_precision = string(sizeof(T))
    nverts = length(verts)
    nverts_str = string(nverts)
    point_dim = length(verts[1])
    if point_dim == 2
        XYZ = "XY"
        dim = " 2"
    else
        error("Invalid point dimension.")
#        XYZ = "XYZ"
#        dim = " 3"
    end
    # Convert the points into an array
    vert_array = zeros(T, point_dim, nverts)
    for i in eachindex(verts)
        vert_array[1, i] = verts[i][1]
        vert_array[2, i] = verts[i][2]
    end
    # Geometry
    xgeom = ElementNode("Geometry")
    link!(xml, xgeom)
    link!(xgeom, AttributeNode("GeometryType", XYZ))
    # DataItem
    xdataitem = ElementNode("DataItem")
    link!(xgeom, xdataitem)
    link!(xdataitem, AttributeNode("DataType", "Float"))
    link!(xdataitem, AttributeNode("Dimensions", nverts_str * dim))
    link!(xdataitem, AttributeNode("Format", "HDF"))
    link!(xdataitem, AttributeNode("Precision", float_precision))
    h5_text_item = HDF5.name(h5_mesh)
    link!(xdataitem, TextNode(string(h5_filepath, ":", h5_text_item, "/vertices")))
    # Write the h5
    h5_mesh["vertices", compress = 3] = vert_array
    return nothing
end

function _write_xdmf_topology!(xml::EzXML.Node,
                               h5_filepath::String,
                               h5_mesh::HDF5.Group,
                               mesh::MeshFile{T, I}) where {T <: AbstractFloat, I <: Integer}
    nel = length(mesh.element_types)
    nelements_str = string(nel)
    topo_length = nel + length(mesh.elements)
    int_precision = string(sizeof(I))
    topo_array = Vector{I}(undef, topo_length)
    topo_ctr = 1
    for i in 1:nel
        eltype = mesh.element_types[i]
        topo_array[topo_ctr] = eltype
        npts = points_in_vtk_type(xdmf2vtk(eltype))
        for j in 1:npts
            offset = mesh.element_offsets[i]
            topo_array[topo_ctr + j] = mesh.elements[offset + j - 1] - I(1)
        end
        topo_ctr += npts + 1
    end
    ndims = string(length(topo_array))
    # Topology
    xtopo = ElementNode("Topology")
    link!(xml, xtopo)
    link!(xtopo, AttributeNode("TopologyType", "Mixed"))
    link!(xtopo, AttributeNode("NumberOfElements", nelements_str))
    # DataItem
    xdataitem = ElementNode("DataItem")
    link!(xtopo, xdataitem)
    link!(xdataitem, AttributeNode("DataType", "Int"))
    link!(xdataitem, AttributeNode("Dimensions", ndims))
    link!(xdataitem, AttributeNode("Format", "HDF"))
    link!(xdataitem, AttributeNode("Precision", int_precision))
    h5_text_item = HDF5.name(h5_mesh)
    link!(xdataitem, TextNode(string(h5_filepath, ":", h5_text_item, "/connectivity")))
    # Write the h5
    h5_mesh["connectivity", compress = 3] = topo_array
    return nothing
end

function _write_xdmf_materials!(xml::EzXML.Node,
                                h5_filepath::String,
                                h5_mesh::HDF5.Group,
                                mesh::MeshFile{T, I}) where {T <: AbstractFloat, I <: Integer}
    nel = length(mesh.element_types)
    material_array = zeros(Int8, nel)
    # Material names
    material_names = String[]
    for elset_name in keys(mesh.elsets)
        if startswith(elset_name, "Material:")
            push!(material_names, elset_name)
        end
    end
    sort!(material_names)
    # Initialize the material array
    for (i, elset_name) in enumerate(material_names)
        for el in mesh.elsets[elset_name]
            if material_array[el] != 0
                error("Element " * string(el) * " is in multiple materials.")
            end
            material_array[el] = Int8(i)
        end
    end
    int_precision = string(sizeof(Int8))

    # Material
    xmaterial = ElementNode("Attribute")
    link!(xml, xmaterial)
    link!(xmaterial, AttributeNode("Center", "Cell"))
    link!(xmaterial, AttributeNode("Name", "Material"))
    # DataItem
    xdataitem = ElementNode("DataItem")
    link!(xmaterial, xdataitem)
    link!(xdataitem, AttributeNode("DataType", "Int"))
    link!(xdataitem, AttributeNode("Dimensions", string(nel)))
    link!(xdataitem, AttributeNode("Format", "HDF"))
    link!(xdataitem, AttributeNode("Precision", int_precision))
    h5_text_item = HDF5.name(h5_mesh)
    link!(xdataitem, TextNode(string(h5_filepath, ":", h5_text_item, "/material")))
    # Write the h5
    h5_mesh["material", compress = 3] = material_array .- Int8(1) # 0-based indexing
    return nothing
end

function _write_xdmf_groups!(xml::EzXML.Node,
                             h5_filepath::String,
                             h5_mesh::HDF5.Group,
                             mesh::MeshFile{T, I}) where {T <: AbstractFloat, I <: Integer}
    for set_name in keys(mesh.elsets)
        if startswith(set_name, "Material:")
            continue
        end
        id_array = collect(mesh.elsets[set_name]) .- I(1)
        int_precision = string(sizeof(I))
        # Set
        xset = ElementNode("Set")
        link!(xml, xset)
        link!(xset, AttributeNode("Name", set_name))
        link!(xset, AttributeNode("SetType", "Cell"))
        # DataItem
        xdataitem = ElementNode("DataItem")
        link!(xset, xdataitem)
        link!(xdataitem, AttributeNode("DataType", "Int"))
        nelems = string(length(id_array))
        link!(xdataitem, AttributeNode("Dimensions", nelems))
        link!(xdataitem, AttributeNode("Format", "HDF"))
        link!(xdataitem, AttributeNode("Precision", int_precision))
        h5_text_item = HDF5.name(h5_mesh)
        link!(xdataitem, TextNode(string(h5_filepath, ":", h5_text_item, "/", set_name)))
        # Write the h5
        h5_mesh[set_name, compress = 3] = id_array
    end
    return nothing
end

# ------------------------------------------------------------------------------

function write_xdmf_file!(mesh::MeshFile{T, I}
    ) where {T <:  AbstractFloat, I <: Integer}
    @info "Writing XDMF file " * mesh.filename
    # Check valid filepath



    if mesh.format == XDMF_FORMAT
        # Do nothing
    elseif mesh.format == ABAQUS_FORMAT
        # Convert to XDMF by changing the numerical 
        # values of the element types. 
        map!(vtk2xdmf, mesh.element_types, mesh.element_types) 
        mesh.format = XDMF_FORMAT
        # Change the filename
        mesh.filename = replace(mesh.filename, ".inp" => ".xdmf")
    else
        error("Invalid mesh format.")
    end

    if !endswith(mesh.filepath, ".xdmf")
        error("Invalid filepath.")
    end
    # h5
    h5_filepath = mesh.filepath[1:(end - 4)] * "h5"
    h5_file = h5open(h5_filepath, "w")
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
        material_names = String[]
        for elset_name in keys(mesh.elsets)
            if startswith(elset_name, "Material:")
                push!(material_names, elset_name[11:end])
            end
        end
        sort!(material_names)
        if 0 < length(material_names)
            xmaterials = ElementNode("Information")
            link!(xdomain, xmaterials)
            link!(xmaterials, AttributeNode("Name", "Materials"))
            link!(xmaterials, TextNode(join(material_names, " ")))
        end
        # Add uniform grid
        _add_uniform_grid_xdmf!(xdomain, h5_filepath, h5_file, mesh)
    finally
        write(mesh.filepath, xdoc)
        close(h5_file)
    end
    return nothing
end

##################################################################################
##                                    READ
##################################################################################
xdmf_read_error() = error("Error while reading XDMF file.")

# Helper functions for read_xdmf
# -------------------------------------------------------------------------------------------------
#function _read_xdmf_uniform_grid(xgrid::EzXML.Node,
#                                 h5_file::HDF5.File,
#                                 material_names::Vector{String})
#    T = UM2_MESH_FLOAT_TYPE         
#    I = UM2_MESH_INT_TYPE
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
#    if dim != 2
#        error("Only 2D meshes are supported.")
#    end
#    float_type = eltype(points_xyz)
#    points = collect(reinterpret(reshape, Point{dim, float_type}, points_xyz))
#    # Connectivity
#    connectivity = read(h5_file[connectivity_path])
#    # count elements
#    conn_length = length(connectivity)
#    nelements = 0
#    offset = 1
#    while offset < conn_length
#        nelements += 1
#        xdmf_type = Int8(connectivity[offset])
#        vtk_type = xdmf2vtk(xdmf_type)
#        offset += points_in_vtk_type(vtk_type) + 1
#    end
#    # set offsets and types
#    offsets = Vector{I}(undef, nelements + 1)
#    element_types = Vector{Int8}(undef, nelements)
#    offset = 1
#    for i in 1:nelements
#        offsets[i] = offset
#        xdmf_type = Int8(connectivity[offset])
#        element_types[i] = xdmf_type
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
#    materials = zeros(Int8, nelements)
#    if material_path != ""
#        materials[:] = read(h5_file[material_path]) .+= 1
#    end
#    # Groups
#    groups = Dict{String, Set{I}}()
#    for i in 1:length(group_paths)
#        groups[group_names[i]] = Set{I}(read(h5_file[group_paths[i]]) .+= 1)
#    end
#    # Convert the materials to groups
#    for i in 1:length(material_names)
#        groups["Material: " * material_names[i]] = Set{I}(findall(x -> x == i, materials))
#    end
#    return MeshFile{T, I}("", XDMF_FORMAT, name, points, element_types, 
#                          offsets, connectivity, groups) 
#end
##
##function _setup_xdmf_tree!(xmlnode::EzXML.Node,
##                          treenode::Tree{Tuple{Int64, String}},
##                          ids::Vector{Int64},
##                          level::Int64 = 0)
##    level += 1
##    if length(ids) < level
##        push!(ids, 0)
##    end
##    if hasnode(xmlnode)
##        for xmlchild in eachnode(xmlnode)
##            if nodename(xmlchild) == "Grid"
##                ids[level] += 1
##                treechild = Tree((ids[level], xmlchild["Name"]), treenode)
##                _setup_xdmf_tree!(xmlchild, treechild, ids, level)
##            end
##        end
##    end
##    return nothing
##end
##
##function _get_volume_mesh_params_from_xdmf(xgrid::EzXML.Node,
##                                           h5_file::HDF5.File)
##    xmlchild = firstnode(xgrid)
##    max_iters = 5
##    i = 1
##    while i ≤ max_iters
##        if nodename(xmlchild) == "Grid" && xmlchild["GridType"] == "Uniform"
##            points_path = ""
##            connectivity_path = ""
##            for child in eachnode(xmlchild)
##                child_name = nodename(child)
##                m = match(r"(?<=:/).*", nodecontent(child))
##                path = string(m.match)
##                if child_name == "Geometry"
##                    points_path = path
##                elseif child_name == "Topology"
##                    connectivity_path = string(m.match)
##                end
##                if points_path != "" && connectivity_path != ""
##                    break
##                end
##            end
##            # Points
##            points_xyz = read(h5_file[points_path])
##            dim, npoints = size(points_xyz)
##            float_type = eltype(points_xyz)
##            # Connectivity
##            connectivity = read(h5_file[connectivity_path])
##            uint_type = eltype(connectivity)
##            return (dim, float_type, uint_type)
##        end
##        xmlchild = firstnode(xmlchild)
##        i += 1
##    end
##    return error("Could not determine volume mesh parameters.")
##end
##
##function _setup_xdmf_leaf_meshes!(xmlnode::EzXML.Node,
##                                  h5_file::HDF5.File,
##                                  idx::Int64,
##                                  leaf_meshes::Vector{<:VolumeMesh},
##                                  material_names::Vector{String})
##    id = idx
##    if hasnode(xmlnode)
##        for xmlchild in eachnode(xmlnode)
##            if nodename(xmlchild) == "Grid"
##                if xmlchild["GridType"] == "Uniform"
##                    leaf_meshes[id] = _read_xdmf_uniform_grid(xmlchild, h5_file,
##                                                              material_names)
##                    id += 1
##                elseif xmlchild["GridType"] == "Tree"
##                    id = _setup_xdmf_leaf_meshes!(xmlchild, h5_file, id,
##                                                  leaf_meshes, material_names)
##                else
##                    error("Unsupported GridType")
##                end
##            end
##        end
##    end
##    return id
##end
##
#
## ------------------------------------------------------------------------ #
#
#function read_xdmf_file(filepath::String)
#    @info "Reading XDMF file: " * filepath
#    xdoc = readxml(filepath)
#    xroot = root(xdoc)
#    nodename(xroot) != "Xdmf" && xdmf_read_error()
#    h5_file = h5open(filepath[begin:(end - 4)] * "h5", "r")
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
#            mesh_file = _read_xdmf_uniform_grid(xgrid, h5_file, material_names)
#            mesh_file.filepath = filepath
#            return mesh_file
###        elseif grid_type == "Tree"
###            # Create tree
###            root = Tree((1, xgrid["Name"]))
###            _setup_xdmf_tree!(xgrid, root, [0])
###            nleaf_meshes = nleaves(root)
###            dim, float_type, uint_type = _get_volume_mesh_params_from_xdmf(xgrid, h5_file)
###            leaf_meshes = Vector{VolumeMesh{dim, float_type, uint_type}}(undef,
###                                                                         nleaf_meshes)
###            # fill the leaf meshes
###            nleaf = _setup_xdmf_leaf_meshes!(xgrid, h5_file, 1, leaf_meshes, material_names)
###            @assert nleaf - 1 == nleaf_meshes
###            return MeshPartitionTree(root, leaf_meshes)
#        else
#            xdmf_read_error()
#        end
#    finally
#        close(h5_file)
#    end
#    return nothing
#end
