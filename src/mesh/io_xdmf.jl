const XDMF_TRIANGLE = 4 
const XDMF_QUAD = 5 
const XDMF_QUADRATIC_TRIANGLE = 36
const XDMF_QUADRATIC_QUAD = 37
const XDMF_TETRA = 6
const XDMF_HEXAHEDRON = 9
const XDMF_QUADRATIC_TETRA = 38
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
#################################################################################
#                                    WRITE
#################################################################################
function write_xdmf(filename::String, mpt::MeshPartitionTree)
    # Check valid filename
    if !endswith(filename, ".xdmf")
        error("Invalid filename.")
    end

    material_groups = String[]
    for mesh in leaf_meshes(mpt) 
        for group_name in keys(mesh.groups)
            if startswith(group_name, "Material") && group_name ∉ material_groups
                push!(material_groups, group_name)
            end
        end
    end
    sort!(material_groups)
    # h5 filename
    h5_filename = filename[1:end-4]*"h5"
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
        materials = [ mat[11:end] for mat in material_groups ]
        if materials != String[] 
            xmaterials = ElementNode("Information")                         
            link!(xdomain, xmaterials)
            link!(xmaterials, AttributeNode("Name", "Materials"))
            link!(xmaterials, TextNode(join(materials, " ")))
        end
        _add_mesh_partition_xdmf!(xdomain, h5_filename, h5_file, 
                                  mpt.partition_tree, mpt.leaf_meshes, material_groups)
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
    h5_filename = filename[1:end-4]*"h5"
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
        
        material_groups = sort!(filter!(x->startswith(x, "Material"), 
                                        collect(keys(mesh.groups))
                                       )
                               )
        materials = [ mat[11:end] for mat in material_groups ]

        # Material names
        if materials != String[] 
            xmaterials = ElementNode("Information")
            link!(xdomain, xmaterials)
            link!(xmaterials, AttributeNode("Name", "Materials"))
            link!(xmaterials, TextNode(join(materials, " ")))
        end

    # Add uniform grid
    _add_uniform_grid_xdmf!(xdomain, h5_filename, h5_file, mesh, material_groups)

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
                                mesh::AbstractMesh,
                                materials::Vector{String})
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
    if mesh.groups != Dict{String,BitSet}() 
        _write_xdmf_groups!(xgrid, h5_filename, h5_group, mesh)
    end
 
    # Materials
    if materials != String[] 
        _write_xdmf_materials!(xgrid, h5_filename, h5_group, mesh, materials)
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
        XYZ= "XYZ"
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
    link!(xdataitem, AttributeNode("Dimensions", nverts_str*dim))
    link!(xdataitem, AttributeNode("Format", "HDF"))
    link!(xdataitem, AttributeNode("Precision", float_precision))
    h5_text_item = split(string(h5_mesh))[2]
    link!(xdataitem, TextNode(string(h5_filename, ":", h5_text_item, "/points")))
    # Write the h5
    h5_mesh["points", compress = 3] = vert_array
    return nothing
end

_nelements_xdmf(mesh::VolumeMesh) = length(mesh.types)
_nelements_xdmf(mesh::PolytopeVertexMesh) = length(mesh.polytopes)

_topo_length_xdmf(mesh::VolumeMesh) = length(mesh.types) + length(mesh.connectivity)
function _topo_length_xdmf(mesh::PolytopeVertexMesh)
    return mapreduce(x->length(vertices(x)), +, polytopes(mesh)) + length(mesh.polytopes)
end

_uint_type_xdmf(mesh::VolumeMesh{Dim,T,U}) where {Dim,T,U} = U
_uint_type_xdmf(mesh::PolytopeVertexMesh) = typeof(mesh.polytopes[1][1])

function _populate_topo_array_xdmf!(topo_array, mesh::VolumeMesh)
    topo_ctr = 1
    for i in eachindex(mesh.types)
        topo_array[topo_ctr] = vtk2xdmf(mesh.types[i])
        topo_ctr += 1
        len_element = points_in_vtk_type(mesh.types[i])
        offset = mesh.offsets[i]
        # adjust 1-based to 0-based indexing
        @. topo_array[topo_ctr:topo_ctr + len_element - 1] = 
            mesh.connectivity[offset:offset + len_element - 1] - 1
        topo_ctr += len_element
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
        topo_array[topo_ctr:topo_ctr + len_element - 1] .= vertices(p) .- 1 
        topo_ctr += len_element
    end
    return nothing
end
    
function _write_xdmf_topology!(xml::EzXML.Node, 
                              h5_filename::String, 
                              h5_mesh::HDF5.Group, 
                              mesh::AbstractMesh)
    nelements = _nelements_xdmf(mesh) 
    nelements_str = string(nelements)
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
    h5_text_item = split(string(h5_mesh))[2]
    link!(xdataitem, TextNode(string(h5_filename, ":", h5_text_item, "/connectivity")))
    # Write the h5
    h5_mesh["connectivity", compress = 3] = topo_array
    return nothing
end

function _write_xdmf_materials!(xml::EzXML.Node, 
                               h5_filename::String, 
                               h5_mesh::HDF5.Group, 
                               mesh::AbstractMesh,
                               materials::Vector{String})
    nelements = _nelements_xdmf(mesh) 
    mat_id_array = fill(-1, nelements)
    mesh_group_names = keys(mesh.groups)
    for (mat_id, material_name) in enumerate(materials)
        if material_name in mesh_group_names 
            for cell in mesh.groups[material_name]
                if mat_id_array[cell] === -1
                    mat_id_array[cell] = mat_id - 1 # 0 based indexing
                else
                    error("Mesh cell "*string(cell)*" has multiple materials assigned to it.")
                end
            end
        end
    end
    if any(x->x === -1, mat_id_array)
        error("Some mesh cells do not have a material.")
    end
    N = length(materials)
    uint_type = _select_uint_type(N)
    uint_precision= string(sizeof(uint_type)) 

    # MaterialID
    xmaterial = ElementNode("Attribute")
    link!(xml, xmaterial)
    link!(xmaterial, AttributeNode("Center", "Cell"))
    link!(xmaterial, AttributeNode("Name", "Material"))
    # DataItem
    xdataitem = ElementNode("DataItem")
    link!(xmaterial, xdataitem)
    link!(xdataitem, AttributeNode("DataType", "UInt"))
    link!(xdataitem, AttributeNode("Dimensions", string(nelements)))
    link!(xdataitem, AttributeNode("Format", "HDF"))
    link!(xdataitem, AttributeNode("Precision", uint_precision))
    h5_text_item = split(string(h5_mesh))[2]
    link!(xdataitem, TextNode(string(h5_filename, ":", h5_text_item, "/material")))
    # Write the h5
    h5_mesh["material", compress = 3] = mat_id_array
    return nothing
end

function _write_xdmf_groups!(xml::EzXML.Node, 
                             h5_filename::String, 
                             h5_mesh::HDF5.Group, 
                             mesh::AbstractMesh)
    for set_name in keys(mesh.groups)
        if startswith(set_name, "Material")
            continue
        end
        grp = mesh.groups[set_name]
        N = maximum(grp)
        uint_type = _select_uint_type(N)
        uint_precision= string(sizeof(uint_type)) 

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
        nelements = string(length(grp))
        link!(xdataitem, AttributeNode("Dimensions", nelements))
        link!(xdataitem, AttributeNode("Format", "HDF"))
        link!(xdataitem, AttributeNode("Precision", uint_precision))
        h5_text_item = split(string(h5_mesh))[2]
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
                                   leaf_meshes::Vector{<:AbstractMesh},
                                   materials::Vector{String})
    id = node.data[1]
    if id === 0 # Internal node
        name = node.data[2]
        # Grid
        xgrid = ElementNode("Grid")
        link!(xml, xgrid)
        link!(xgrid, AttributeNode("Name", name))
        link!(xgrid, AttributeNode("GridType", "Tree"))
        # h5_group
        h5_group = create_group(h5_mesh, name)
        for child in node.children
            _add_mesh_partition_xdmf!(xgrid, h5_filename, h5_group, child, 
                                      leaf_meshes, materials)
        end
    else # Leaf node 
        _add_uniform_grid_xdmf!(xml, h5_filename, h5_mesh, 
                                leaf_meshes[id], materials)
    end
    return nothing
end

#################################################################################
#                                    READ
#################################################################################
#xdmf_read_error(x::String) = error("Error reading XDMF file.")
#function read_xdmf(path::String, ::Type{T}) where {T<:AbstractFloat}
#    xdoc = readxml(path)
#    xroot = root(xdoc)
#    nodename(xroot) != "Xdmf" && xdmf_read_error()
#    try
#        version = xroot["Version"]
#        version != "3.0" && xdmf_read_error()
#        xdomain = firstnode(xroot)
#        nodename(xdomain) != "Domain" && xdmf_read_error()
#        material_names = String[]
#        if 1 < countnodes(xdomain) && nodename(firstnode(xdomain)) == "Information"
#            append!(material_names, 
#        end
#
#    finally
#
#    end
#end
