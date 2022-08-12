const XDMF_TRIANGLE             = 4
const XDMF_QUAD                 = 5
const XDMF_QUADRATIC_TRIANGLE   = 36
const XDMF_QUADRATIC_QUAD       = 37
const XDMF_TETRA                = 6
const XDMF_HEXAHEDRON           = 9
const XDMF_QUADRATIC_TETRA      = 38
const XDMF_QUADRATIC_HEXAHEDRON = 48

function vtk2xdmf(vtk_type)
    if vtk_type == VTK_TRIANGLE
        return XDMF_TRIANGLE
    elseif vtk_type == VTK_QUAD
        return XDMF_QUAD
    elseif vtk_type == VTK_QUADRATIC_TRIANGLE
        return XDMF_QUADRATIC_TRIANGLE
    elseif vtk_type == VTK_QUADRATIC_QUAD
        return XDMF_QUADRATIC_QUAD
    elseif vtk_type == VTK_TETRA
        return XDMF_TETRA
    elseif vtk_type == VTK_HEXAHEDRON
        return XDMF_HEXAHEDRON
    elseif vtk_type == VTK_QUADRATIC_TETRA
        return XDMF_QUADRATIC_TETRA
    elseif vtk_type == VTK_QUADRATIC_HEXAHEDRON
        return XDMF_QUADRATIC_HEXAHEDRON
    else
        error("Invalid VTK type.")
        return nothing
    end
end

function xdmf2vtk(xdmf_type)
    if xdmf_type == XDMF_TRIANGLE
        return VTK_TRIANGLE
    elseif xdmf_type == XDMF_QUAD
        return VTK_QUAD
    elseif xdmf_type == XDMF_QUADRATIC_TRIANGLE
        return VTK_QUADRATIC_TRIANGLE
    elseif xdmf_type == XDMF_QUADRATIC_QUAD
        return VTK_QUADRATIC_QUAD
    elseif xdmf_type == XDMF_TETRA
        return VTK_TETRA
    elseif xdmf_type == XDMF_HEXAHEDRON
        return VTK_HEXAHEDRON
    elseif xdmf_type == XDMF_QUADRATIC_TETRA
        return VTK_QUADRATIC_TETRA
    elseif xdmf_type == XDMF_QUADRATIC_HEXAHEDRON
        return VTK_QUADRATIC_HEXAHEDRON
    else
        error("Invalid XDMF type.")
        return nothing
    end
end

#################################################################################
#                                    WRITE
#################################################################################
function write_xdmf(filename::String, mpt::MeshPartitionTree)
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
        if 0 < length(mpt.leaf_meshes[1].material_names)
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

function write_xdmf(filename::String, mesh::AbstractMesh)
    # Check valid filename
    if !endswith(filename, ".xdmf")
        error("Invalid filename.")
    end
    # h5
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
        if 0 < length(mesh.material_names)
            xmaterials = ElementNode("Information")
            link!(xdomain, xmaterials)
            link!(xmaterials, AttributeNode("Name", "Materials"))
            link!(xmaterials, TextNode(join(mesh.material_names, " ")))
        end
        # Add uniform grid
        _add_uniform_grid_xdmf!(xdomain, h5_filename, h5_file, mesh)
    finally
        write(filename, xdoc)
        close(h5_file)
    end
    return nothing
end

# Helper functions for write_xdmf
# -------------------------------------------------------------------------------------------------
function _add_uniform_grid_xdmf!(xml::EzXML.Node,
                                 h5_filename::String,
                                 h5_mesh::Union{HDF5.Group, HDF5.File},
                                 mesh::AbstractMesh)
    # Grid                                                                       
    xgrid = ElementNode("Grid")
    link!(xml, xgrid)
    link!(xgrid, AttributeNode("Name", mesh.name))
    link!(xgrid, AttributeNode("GridType", "Uniform"))
    # h5
    h5_group = create_group(h5_mesh, mesh.name)
    # Geometry
    _write_xdmf_geometry!(xgrid, h5_filename, h5_group, mesh)
    # Topology
    _write_xdmf_topology!(xgrid, h5_filename, h5_group, mesh)
    # Non-material groups 
    if 0 < length(mesh.groups)
        _write_xdmf_groups!(xgrid, h5_filename, h5_group, mesh)
    end
    # Materials
    if 0 < length(mesh.material_names)
        _write_xdmf_materials!(xgrid, h5_filename, h5_group, mesh)
    end
    return nothing
end

function _write_xdmf_geometry!(xml::EzXML.Node,
                               h5_filename::String,
                               h5_mesh::HDF5.Group,
                               mesh::AbstractMesh)
    verts = points(mesh)
    float_type = typeof(verts[1][1])
    float_precision = string(sizeof(float_type))
    nverts = length(verts)
    nverts_str = string(nverts)
    point_dim = length(verts[1])
    if point_dim == 2
        XYZ = "XY"
        dim = " 2"
    else
        XYZ = "XYZ"
        dim = " 3"
    end
    # Convert the points into an array
    vert_array = zeros(float_type, point_dim, nverts)
    for i in eachindex(verts)
        vert_array[:, i] = coordinates(verts[i])
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
    link!(xdataitem, TextNode(string(h5_filename, ":", h5_text_item, "/points")))
    # Write the h5
    h5_mesh["points", compress = 3] = vert_array
    return nothing
end

_topo_length_xdmf(mesh::VolumeMesh) = nelements(mesh) + length(mesh.connectivity)
function _topo_length_xdmf(mesh::PolytopeVertexMesh)
    return mapreduce(x -> length(vertices(x)), +, polytopes(mesh)) + length(mesh.polytopes)
end

_uint_type_xdmf(mesh::VolumeMesh{D, T, U}) where {D, T, U} = U
_uint_type_xdmf(mesh::PolytopeVertexMesh) = typeof(mesh.polytopes[1][1])

function _populate_topo_array_xdmf!(topo_array, mesh::VolumeMesh{D}) where {D}
    topo_ctr = 1
    for i in 1:nelements(mesh)
        Δ = offset_diff(i, mesh)
        topo_array[topo_ctr] = vtk2xdmf(_volume_mesh_points_to_vtk_type(D, Δ))
        topo_ctr += 1
        offset = mesh.offsets[i]
        # adjust 1-based to 0-based indexing
        @. topo_array[topo_ctr:(topo_ctr + Δ - 1)] = mesh.connectivity[offset:(offset + Δ - 1)] -
                                                     1
        topo_ctr += Δ
    end
    return nothing
end

function _populate_topo_array_xdmf!(topo_array, mesh::PolytopeVertexMesh)
    ptopes = mesh.polytopes
    topo_ctr = 1
    for i in eachindex(ptopes)
        p = ptopes[i]
        topo_array[topo_ctr] = vtk2xdmf(vtk_type(typeof(p)))
        topo_ctr += 1
        len_element = length(vertices(p))
        # adjust 1-based to 0-based indexing
        topo_array[topo_ctr:(topo_ctr + len_element - 1)] .= vertices(p) .- 1
        topo_ctr += len_element
    end
    return nothing
end

function _write_xdmf_topology!(xml::EzXML.Node,
                               h5_filename::String,
                               h5_mesh::HDF5.Group,
                               mesh::AbstractMesh)
    nel = nelements(mesh)
    nelements_str = string(nel)
    topo_length = _topo_length_xdmf(mesh)
    uint_type = _uint_type_xdmf(mesh)
    uint_precision = string(sizeof(uint_type))
    topo_array = Vector{uint_type}(undef, topo_length)
    _populate_topo_array_xdmf!(topo_array, mesh)
    ndims = string(length(topo_array))
    # Topology
    xtopo = ElementNode("Topology")
    link!(xml, xtopo)
    link!(xtopo, AttributeNode("TopologyType", "Mixed"))
    link!(xtopo, AttributeNode("NumberOfElements", nelements_str))
    # DataItem
    xdataitem = ElementNode("DataItem")
    link!(xtopo, xdataitem)
    link!(xdataitem, AttributeNode("DataType", "UInt"))
    link!(xdataitem, AttributeNode("Dimensions", ndims))
    link!(xdataitem, AttributeNode("Format", "HDF"))
    link!(xdataitem, AttributeNode("Precision", uint_precision))
    h5_text_item = HDF5.name(h5_mesh)
    link!(xdataitem, TextNode(string(h5_filename, ":", h5_text_item, "/connectivity")))
    # Write the h5
    h5_mesh["connectivity", compress = 3] = topo_array
    return nothing
end

function _write_xdmf_materials!(xml::EzXML.Node,
                                h5_filename::String,
                                h5_mesh::HDF5.Group,
                                mesh::AbstractMesh)
    N = length(mesh.materials)
    uint_precision = string(sizeof(UInt8))

    # Material
    xmaterial = ElementNode("Attribute")
    link!(xml, xmaterial)
    link!(xmaterial, AttributeNode("Center", "Cell"))
    link!(xmaterial, AttributeNode("Name", "Material"))
    # DataItem
    xdataitem = ElementNode("DataItem")
    link!(xmaterial, xdataitem)
    link!(xdataitem, AttributeNode("DataType", "UInt"))
    link!(xdataitem, AttributeNode("Dimensions", string(N)))
    link!(xdataitem, AttributeNode("Format", "HDF"))
    link!(xdataitem, AttributeNode("Precision", uint_precision))
    h5_text_item = HDF5.name(h5_mesh)
    link!(xdataitem, TextNode(string(h5_filename, ":", h5_text_item, "/material")))
    # Write the h5
    h5_mesh["material", compress = 3] = mesh.materials .- 1 # 0-based indexing
    return nothing
end

function _write_xdmf_groups!(xml::EzXML.Node,
                             h5_filename::String,
                             h5_mesh::HDF5.Group,
                             mesh::AbstractMesh)
    for set_name in keys(mesh.groups)
        grp = mesh.groups[set_name]
        N = maximum(grp)
        uint_type = _select_uint_type(N)
        uint_precision = string(sizeof(uint_type))

        id_array = collect(grp) .- 1
        # Set
        xset = ElementNode("Set")
        link!(xml, xset)
        link!(xset, AttributeNode("Name", set_name))
        link!(xset, AttributeNode("SetType", "Cell"))
        # DataItem
        xdataitem = ElementNode("DataItem")
        link!(xset, xdataitem)
        link!(xdataitem, AttributeNode("DataType", "UInt"))
        nelems = string(length(grp))
        link!(xdataitem, AttributeNode("Dimensions", nelems))
        link!(xdataitem, AttributeNode("Format", "HDF"))
        link!(xdataitem, AttributeNode("Precision", uint_precision))
        h5_text_item = HDF5.name(h5_mesh)
        link!(xdataitem, TextNode(string(h5_filename, ":", h5_text_item, "/", set_name)))
        # Write the h5
        h5_mesh[set_name, compress = 3] = id_array
    end
    return nothing
end

function _add_mesh_partition_xdmf!(xml::EzXML.Node,
                                   h5_filename::String,
                                   h5_mesh::Union{HDF5.Group, HDF5.File},
                                   node::Tree{Tuple{Int64, String}},
                                   leaf_meshes::Vector{<:AbstractMesh})
    if !isleaf(node) # Internal node
        name = node.data[2]
        # Grid
        xgrid = ElementNode("Grid")
        link!(xml, xgrid)
        link!(xgrid, AttributeNode("Name", name))
        link!(xgrid, AttributeNode("GridType", "Tree"))
        # h5_group
        h5_group = create_group(h5_mesh, name)
        for child in node.children
            _add_mesh_partition_xdmf!(xgrid, h5_filename, h5_group, child, leaf_meshes)
        end
    else # Leaf node 
        id = node.data[1]
        _add_uniform_grid_xdmf!(xml, h5_filename, h5_mesh, leaf_meshes[id])
    end
    return nothing
end

#################################################################################
#                                    READ
#################################################################################
xdmf_read_error(x::String) = error("Error reading XDMF file.")
function read_xdmf(path::String, ::Type{T}) where {T <: AbstractFloat}
    xdoc = readxml(path)
    xroot = root(xdoc)
    nodename(xroot) != "Xdmf" && xdmf_read_error()
    h5_file = h5open(path[begin:(end - 4)] * "h5", "r")
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
            return _read_xdmf_uniform_grid(xgrid, h5_file, material_names)
        elseif grid_type == "Tree"
            # Create tree
            root = Tree((1, xgrid["Name"]))
            _setup_xdmf_tree!(xgrid, root, [0])
            nleaf_meshes = nleaves(root) 
            dim, float_type, uint_type = _get_volume_mesh_params_from_xdmf(xgrid, h5_file)
            leaf_meshes = Vector{VolumeMesh{dim, float_type, uint_type}}(undef,
                                                                         nleaf_meshes)
            # fill the leaf meshes
            nleaf = _setup_xdmf_leaf_meshes!(xgrid, h5_file, 1, leaf_meshes, material_names)
            @assert nleaf - 1 == nleaf_meshes
            return MeshPartitionTree(root, leaf_meshes)
        else
            xdmf_read_error()
        end
    finally
        close(h5_file)
    end
    return nothing
end

# Helper functions for read_xdmf
# -------------------------------------------------------------------------------------------------
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
            warn("Unused XML node: " * child_name)
        end
    end
    # Points
    name = xgrid["Name"]
    points_xyz = read(h5_file[points_path])
    dim, npoints = size(points_xyz)
    float_type = eltype(points_xyz)
    points = collect(reinterpret(reshape, Point{dim, float_type}, points_xyz))
    # Connectivity
    connectivity = read(h5_file[connectivity_path])
    uint_type = eltype(connectivity)
    # count elements
    conn_length = length(connectivity)
    nelements = 0
    offset = 1
    while offset < conn_length
        nelements += 1
        xdmf_type = connectivity[offset]
        vtk_type = xdmf2vtk(xdmf_type)
        offset += points_in_vtk_type(vtk_type) + 1
    end
    # set offsets and types
    offsets = Vector{uint_type}(undef, nelements + 1)
    offset = 1
    for i in 1:nelements
        offsets[i] = offset
        xdmf_type = connectivity[offset]
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
    materials = zeros(UInt8, nelements)
    if material_path != ""
        materials[:] = read(h5_file[material_path]) .+= 1
    end
    # Groups
    groups = Dict{String, BitSet}()
    for i in 1:length(group_paths)
        groups[group_names[i]] = BitSet(read(h5_file[group_paths[i]]) .+= 1)
    end
    return VolumeMesh{dim, float_type, uint_type}(points, offsets, connectivity,
                                                  materials, material_names, name, groups)
end

function _setup_xdmf_tree!(xmlnode::EzXML.Node,
                          treenode::Tree{Tuple{Int64, String}},
                          ids::Vector{Int64},
                          level::Int64 = 0)
    level += 1
    if length(ids) < level
        push!(ids, 0)
    end
    if hasnode(xmlnode)
        for xmlchild in eachnode(xmlnode)
            if nodename(xmlchild) == "Grid"
                ids[level] += 1
                treechild = Tree((ids[level], xmlchild["Name"]), treenode)
                _setup_xdmf_tree!(xmlchild, treechild, ids, level)
            end
        end
    end
    return nothing
end

function _get_volume_mesh_params_from_xdmf(xgrid::EzXML.Node,
                                           h5_file::HDF5.File)
    xmlchild = firstnode(xgrid)
    max_iters = 5
    i = 1
    while i ≤ max_iters
        if nodename(xmlchild) == "Grid" && xmlchild["GridType"] == "Uniform"
            points_path = ""
            connectivity_path = ""
            for child in eachnode(xmlchild)
                child_name = nodename(child)
                m = match(r"(?<=:/).*", nodecontent(child))
                path = string(m.match)
                if child_name == "Geometry"
                    points_path = path
                elseif child_name == "Topology"
                    connectivity_path = string(m.match)
                end
                if points_path != "" && connectivity_path != ""
                    break
                end
            end
            # Points
            points_xyz = read(h5_file[points_path])
            dim, npoints = size(points_xyz)
            float_type = eltype(points_xyz)
            # Connectivity
            connectivity = read(h5_file[connectivity_path])
            uint_type = eltype(connectivity)
            return (dim, float_type, uint_type)
        end
        xmlchild = firstnode(xmlchild)
        i += 1
    end
    return error("Could not determine volume mesh parameters.")
end

function _setup_xdmf_leaf_meshes!(xmlnode::EzXML.Node,
                                  h5_file::HDF5.File,
                                  idx::Int64,
                                  leaf_meshes::Vector{<:VolumeMesh},
                                  material_names::Vector{String})
    id = idx
    if hasnode(xmlnode)
        for xmlchild in eachnode(xmlnode)
            if nodename(xmlchild) == "Grid"
                if xmlchild["GridType"] == "Uniform"
                    leaf_meshes[id] = _read_xdmf_uniform_grid(xmlchild, h5_file,
                                                              material_names)
                    id += 1
                elseif xmlchild["GridType"] == "Tree"
                    id = _setup_xdmf_leaf_meshes!(xmlchild, h5_file, id,
                                                  leaf_meshes, material_names)
                else
                    error("Unsupported GridType")
                end
            end
        end
    end
    return id
end
