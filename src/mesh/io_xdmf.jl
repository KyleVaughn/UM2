#################################################################################
#                                    WRITE
#################################################################################

# Helper functions for write_xdmf_file
# -------------------------------------------------------------------------------------------------

function _write_xdmf_geometry!(xml::EzXML.Node,
                               h5_filepath::String,
                               h5_mesh::HDF5.Group,
                               mesh::MeshFile)
    verts = mesh.nodes
    float_precision = string(sizeof(UM_F))
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
    vert_array = zeros(UM_F, point_dim, nverts)
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
                               mesh::MeshFile)
    nel = length(mesh.element_types)
    nelements_str = string(nel)
    topo_length = nel + length(mesh.elements)
    int_precision = string(sizeof(UM_I))
    topo_array = Vector{UM_I}(undef, topo_length)
    topo_ctr = 1
    for i in 1:nel
        eltype = mesh.element_types[i]
        topo_array[topo_ctr] = eltype
        npts = points_in_vtk_type(xdmf2vtk(eltype))
        for j in 1:npts
            offset = mesh.element_offsets[i]
            topo_array[topo_ctr + j] = mesh.elements[offset + j - 1] - UM_I(1)
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
                                mesh::MeshFile,
                                material_names::Vector{String})
    nel = length(mesh.element_types)
    material_array = zeros(Int8, nel)
    # Initialize the material array
    for set_name in keys(mesh.elsets)
        if !startswith(set_name, "Material")
            continue
        end
        stripped_name = set_name[11:end]
        mat_id = findfirst(x -> x == stripped_name, material_names)
        if mat_id === nothing
            error("Material name " * stripped_name* " not found in material_names.")
        end
        for el in mesh.elsets[set_name]
            if material_array[el] != 0
                error("Element " * string(el) * " has multiple materials.")
            end
            material_array[el] = Int8(mat_id)
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
                             mesh::MeshFile)
    for set_name in keys(mesh.elsets)
        if startswith(set_name, "Material:")
            continue
        end
        id_array = collect(mesh.elsets[set_name]) .- UM_I(1)
        int_precision = string(sizeof(UM_I))
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

function _add_uniform_grid_xdmf!(xml::EzXML.Node,
                                 h5_filepath::String,
                                 h5_mesh::Union{HDF5.Group, HDF5.File},
                                 mesh::MeshFile,
                                 material_names::Vector{String})
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
    _write_xdmf_materials!(xgrid, h5_filepath, h5_group, mesh, material_names)
    return nothing
end

function _add_tree_xdmf!(xml::EzXML.Node,
                         h5_filename::String,
                         h5_mesh::Union{HDF5.Group, HDF5.File},
                         node::Tree{Tuple{UM_I, String}},
                         leaf_meshes::Vector{MeshFile},
                         material_names::Vector{String})
    if !isleaf(node) # Internal node
        name = getindex(data(node), 2)
        # Grid
        xgrid = ElementNode("Grid")
        link!(xml, xgrid)
        link!(xgrid, AttributeNode("Name", name))
        link!(xgrid, AttributeNode("GridType", "Tree"))
        # h5_group
        h5_group = create_group(h5_mesh, name)
        for child in children(node)
            _add_tree_xdmf!(xgrid, h5_filename, h5_group, child, 
                            leaf_meshes, material_names)
        end
    else # Leaf node 
        id = getindex(data(node), 1)
        _add_uniform_grid_xdmf!(xml, h5_filename, h5_mesh, leaf_meshes[id],
                                material_names)
    end
    return nothing
end

# ------------------------------------------------------------------------------

function write_xdmf_file!(mesh::MeshFile)
    @info "Writing XDMF file '" * mesh.filepath * "'"
    # Check valid filepath
    if mesh.format === XDMF_FORMAT
        # Do nothing
    elseif mesh.format === ABAQUS_FORMAT
        # Convert to XDMF by changing the numerical 
        # values of the element types. 
        map!(vtk2xdmf, mesh.element_types, mesh.element_types) 
        mesh.format = XDMF_FORMAT
        # Change the filepath
        mesh.filepath = mesh.filepath[1:(end - 3)] * "xdmf"
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
        _add_uniform_grid_xdmf!(xdomain, h5_filepath, h5_file, mesh, 
                                material_names)
    finally
        write(mesh.filepath, xdoc)
        close(h5_file)
    end
    return nothing
end

function write_xdmf_file!(hmf::HierarchicalMeshFile)
    @info "Writing hierarchical XDMF file " * hmf.filepath
    # Check valid filepath
    if hmf.format !== XDMF_FORMAT
        error("Invalid mesh format.")
    end

    if !endswith(hmf.filepath, ".xdmf")
        error("Invalid filepath.")
    end

    # h5
    h5_filepath = hmf.filepath[1:(end - 4)] * "h5"
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
        for mf in hmf.leaf_meshes
            for elset_name in keys(mf.elsets)
                if startswith(elset_name, "Material:") && 
                        elset_name[11:end] ∉ material_names
                    push!(material_names, elset_name[11:end])
                end
            end
        end
        sort!(material_names)
        if 0 < length(material_names)
            xmaterials = ElementNode("Information")
            link!(xdomain, xmaterials)
            link!(xmaterials, AttributeNode("Name", "Materials"))
            link!(xmaterials, TextNode(join(material_names, " ")))
        end
        # Add tree 
        _add_tree_xdmf!(xdomain, h5_filepath, h5_file, 
                        hmf.partition_tree, hmf.leaf_meshes,
                        material_names)
    finally
        write(hmf.filepath, xdoc)
        close(h5_file)
    end
    return nothing
end
##################################################################################
##                                    READ
##################################################################################
xdmf_read_error() = error("Error while reading XDMF file.")

function _read_xdmf_uniform_grid(xgrid::EzXML.Node,
                                 h5_file::HDF5.File,
                                 material_names::Vector{String})
    # Get all the h5 file paths to relevant data
    points_path = ""
    connectivity_path = ""
    material_path = ""
    group_paths = String[]
    group_names = String[]
    for child in eachnode(xgrid)
        child_name = nodename(child)
        m = match(r"(?<=:/).*", nodecontent(child))
        path = string(m.match)
        if child_name == "Geometry"
            points_path = path
        elseif child_name == "Topology"
            connectivity_path = string(m.match)
        elseif child_name == "Attribute" && haskey(child, "Name") &&
               child["Name"] == "Material"
            material_path = path
        elseif child_name == "Set" && haskey(child, "SetType") && child["SetType"] == "Cell"
            push!(group_paths, path)
            push!(group_names, child["Name"])
        else
            @warn "Unused XML node: " * child_name
        end
    end
    # Points
    name = xgrid["Name"]
    points_xyz = read(h5_file[points_path])
    dim, npoints = size(points_xyz)
    if dim != 2
        error("Only 2D meshes are supported.")
    end
    points = Vector{Point2{UM_F}}(undef, npoints)
    for i = 1:npoints
        points[i] = Point2{UM_F}(points_xyz[1, i], points_xyz[2, i])
    end
    # Connectivity
    connectivity = read(h5_file[connectivity_path])
    # count elements
    conn_length = length(connectivity)
    nelements = 0
    offset = 1
    while offset < conn_length
        nelements += 1
        xdmf_type = Int8(connectivity[offset])
        vtk_type = xdmf2vtk(xdmf_type)
        offset += points_in_vtk_type(vtk_type) + 1
    end
    # set offsets and types
    offsets = Vector{UM_I}(undef, nelements + 1)
    element_types = Vector{Int8}(undef, nelements)
    offset = 1
    for i in 1:nelements
        offsets[i] = offset
        xdmf_type = Int8(connectivity[offset])
        element_types[i] = xdmf_type
        vtk_type = xdmf2vtk(xdmf_type)
        offset += points_in_vtk_type(vtk_type) + 1
    end
    deleteat!(connectivity, view(offsets, 1:nelements))
    connectivity .+= 1 # convert 0-based to 1-based
    # Account for deletion
    for i in 1:nelements
        offsets[i] -= i - 1
    end
    # Add the final offset
    offsets[nelements + 1] = length(connectivity) + 1
    # Materials
    materials = zeros(Int8, nelements)
    if material_path != ""
        materials[:] = read(h5_file[material_path]) .+= 1
    end
    # Groups
    groups = Dict{String, Set{UM_I}}()
    for i in 1:length(group_paths)
        groups[group_names[i]] = Set{UM_I}(read(h5_file[group_paths[i]]) .+= 1)
    end
    # Convert the materials to groups
    for i in 1:length(material_names)
        groups["Material:_" * material_names[i]] = Set{UM_I}(findall(x -> x == i, materials))
    end
    return materials, MeshFile("", XDMF_FORMAT, name, points, element_types, 
                                offsets, connectivity, groups) 
end

function _setup_xdmf_tree!(xmlnode::EzXML.Node,
                          node::Tree{Tuple{UM_I, String}},
                          node_ctr::Vector{UM_I},
                          level::UM_I)
    level += UM_I(1)
    if length(node_ctr) < level
        push!(node_ctr, UM_I(0))
    end
    node_ctr[level] += UM_I(1)
    node.data = (node_ctr[level], xmlnode["Name"])
    if hasnode(xmlnode)
        for xmlchild in eachnode(xmlnode)
            if nodename(xmlchild) == "Grid"
                child_node = Tree((UM_I(0), xmlchild["Name"]), node)
                _setup_xdmf_tree!(xmlchild, child_node, node_ctr, level)
            end
        end
    end
    return nothing
end

function _setup_xdmf_leaf_meshes!(xmlnode::EzXML.Node,
                                  h5_file::HDF5.File,
                                  leaf_meshes::Vector{MeshFile},
                                  leaf_materials::Vector{Vector{Int8}},
                                  material_names::Vector{String},
                                  idx::Int64)
    id = idx
    if hasnode(xmlnode)
        for xmlchild in eachnode(xmlnode)
            if nodename(xmlchild) == "Grid"
                if xmlchild["GridType"] == "Uniform"
                    (leaf_materials[id], 
                     leaf_meshes[id]) = _read_xdmf_uniform_grid(xmlchild, 
                                                                h5_file,
                                                                material_names)
                    id += 1
                elseif xmlchild["GridType"] == "Tree"
                    id = _setup_xdmf_leaf_meshes!(xmlchild, h5_file,
                                                  leaf_meshes, leaf_materials, 
                                                  material_names, id)
                else
                    error("Unsupported GridType")
                end
            end
        end
    end
    return id
end

# ------------------------------------------------------------------------ #

# Not type-stable, since it may return a MeshFile or HierarchicalMeshFile
function read_xdmf_file(filepath::String)
    @info "Reading XDMF file: '" * filepath * "'"
    xdoc = readxml(filepath)
    xroot = root(xdoc)
    nodename(xroot) != "Xdmf" && xdmf_read_error()
    h5_file = h5open(filepath[begin:(end - 4)] * "h5", "r")
    try
        version = xroot["Version"]
        version != "3.0" && xdmf_read_error()
        xdomain = firstnode(xroot)
        nodename(xdomain) != "Domain" && xdmf_read_error()
        material_names = String[]
        nnodes = countnodes(xdomain)
        if 2 == nnodes && nodename(firstnode(xdomain)) == "Information"
            append!(material_names, split(nodecontent(firstnode(xdomain)), " "))
            xgrid = nodes(xdomain)[2]
        elseif 1 == nnodes
            xgrid = firstnode(xdomain)
        else
            xdmf_read_error()
        end
        grid_type = xgrid["GridType"]
        if grid_type == "Uniform"
            materials, mesh_file = _read_xdmf_uniform_grid(xgrid, h5_file, material_names)
            mesh_file.filepath = filepath
            return material_names, materials, mesh_file
        elseif grid_type == "Tree"
            @info "... Mesh is hierarchical"
            # Create tree
            root = Tree((UM_I(0), xgrid["Name"]))
            _setup_xdmf_tree!(xgrid, root, [UM_I(0)], UM_I(0))
            nleaf_meshes = num_leaves(root)
            leaf_meshes = Vector{MeshFile}(undef, nleaf_meshes)
            leaf_materials = Vector{Vector{Int8}}(undef, nleaf_meshes)
            # fill the leaf meshes
            nleaf = _setup_xdmf_leaf_meshes!(xgrid, h5_file, leaf_meshes, 
                                             leaf_materials, material_names, 1)
            @assert nleaf - 1 == nleaf_meshes
            return material_names, leaf_materials, HierarchicalMeshFile(filepath, XDMF_FORMAT, root, leaf_meshes)
        else
            xdmf_read_error()
        end
    finally
        close(h5_file)
    end
    return nothing
end
