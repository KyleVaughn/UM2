function write_xdmf_2d(filename::String, mesh::UnstructuredMesh_2D)
    # h5 filename
    h5_filename = replace(filename, ("xdmf" => "h5"))
    h5_file = h5open(h5_filename, "w")
    h5_mesh = create_group(h5_file, mesh.name)
    # XML
    xdoc = XMLDocument()
    # Xdmf
    xroot = create_root(xdoc, "Xdmf")
    set_attribute(xroot, "Version", "3.0")
    # Domain
    xdomain = new_child(xroot, "Domain")
    # Grid
    xgrid = new_child(xdomain, "Grid")
    set_attribute(xgrid, "Name", mesh.name)
    set_attribute(xgrid, "GridType", "Uniform")

    # Geometry
    _write_xdmf_geometry(xgrid, h5_filename, h5_mesh, mesh)

    # Topology
    _write_xdmf_topology(xgrid, h5_filename, h5_mesh, mesh)

    print(xdoc)
    close(h5_file)
end

function _write_xdmf_geometry(xml::XMLElement, 
                              h5_filename::String, 
                              h5_mesh::HDF5.Group, 
                              mesh::UnstructuredMesh_2D)
    # Geometry
    xgeom = new_child(xml, "Geometry")
    set_attribute(xgeom, "GeometryType", "XYZ")
    # DataItem
    xdataitem = new_child(xgeom, "DataItem")
    set_attribute(xdataitem, "DataType", "Float")
    npoints = length(mesh.points)
    set_attribute(xdataitem, "Dimensions", "$npoints 3")
    set_attribute(xdataitem, "Format", "HDF")
    float_precision = 8
    float_type = Float64
    if typeof(mesh.points[1]) == Point_2D{Float32}
        float_precision = 4
        float_type = Float32
    end
    set_attribute(xdataitem, "Precision", "$float_precision")
    add_text(xdataitem, string(h5_filename, ":/", mesh.name, "/points"))
    # Convert the points into an array
    point_array = zeros(float_type, npoints, 3)
    for i = 1:npoints
        point_array[i, 1] = mesh.points[i][1]
        point_array[i, 2] = mesh.points[i][2]
    end
    point_array = copy(transpose(point_array))
    # Write the h5
    h5_mesh["points"] = point_array
end

function _write_xdmf_topology(xml::XMLElement, 
                              h5_filename::String, 
                              h5_mesh::HDF5.Group, 
                              mesh::UnstructuredMesh_2D)

end
