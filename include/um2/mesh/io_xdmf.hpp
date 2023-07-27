#pragma once

#include <um2/common/Log.hpp>
#include <um2/config.hpp>
// #include <um2/common/sto.hpp>
// #include <um2/mesh/cell_type.hpp>
#include <um2/mesh/MeshFile.hpp>
//
// #include <thrust/logical.h> // thrust::all_of
// #include <thrust/transform.h> // thrust::transform
//
#include <algorithm> // std::transform
#include <string>
#include <vector>
// #include <concepts> // std::floating_point, std::signed_integral
// #include <string.h> // strcmp, strcpy
// #include <fstream> // std::ifstream
// #include <limits> // std::numeric_limits
// #include <string> // std::string

// External dependencies
#include <H5Cpp.h>     // H5::H5File, H5::DataSet, H5::DataSpace, H5::DataType
#include <pugixml.hpp> // pugi::xml_document, pugi::xml_node
                       // H5::Group, H5::PredType, H5::hsize_t

namespace um2
{

// -----------------------------------------------------------------------------
// XDMF mesh file
// -----------------------------------------------------------------------------
// IO for XDMF mesh files.

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void writeXDMFGeometry(pugi::xml_node & xgrid, H5::Group & h5group,
                              std::string const & h5filename, std::string const & h5path,
                              MeshFile<T, I> const & mesh)
{
  Log::debug("Writing XDMF geometry");
  size_t const num_nodes = mesh.nodes_x.size();
  size_t dim = 2;
  if (!mesh.nodes_z.empty()) {
    dim = 3;
  }
  // Create XDMF Geometry node
  auto xgeom = xgrid.append_child("Geometry");
  if (dim == 3) {
    xgeom.append_attribute("GeometryType") = "XYZ";
  } else { // (dim == 2)
    xgeom.append_attribute("GeometryType") = "XY";
  }

  // Create XDMF DataItem node
  auto xdata = xgeom.append_child("DataItem");
  xdata.append_attribute("DataType") = "Float";
  xdata.append_attribute("Dimensions") =
      (std::to_string(num_nodes) + " " + std::to_string(dim)).c_str();
  if constexpr (sizeof(T) == 4) {
    xdata.append_attribute("Precision") = 4;
  } else if constexpr (sizeof(T) == 8) {
    xdata.append_attribute("Precision") = 8;
  }
  xdata.append_attribute("Format") = "HDF";
  std::string const h5geompath = h5filename + ":" + h5path + "/Geometry";
  xdata.append_child(pugi::node_pcdata).set_value(h5geompath.c_str());

  // Create HDF5 data space
  hsize_t dims[2] = {static_cast<hsize_t>(num_nodes), dim};
  H5::DataSpace const h5space(2, dims);
  // Create HDF5 data type
  H5::DataType h5type;
  // NOLINTBEGIN(cppcoreguidelines-slicing)
  if constexpr (sizeof(T) == 4) {
    h5type = H5::PredType::NATIVE_FLOAT;
  } else if constexpr (sizeof(T) == 8) {
    h5type = H5::PredType::NATIVE_DOUBLE;
  }
  // NOLINTEND(cppcoreguidelines-slicing)
  // Create HDF5 data set
  H5::DataSet const h5dataset = h5group.createDataSet("Geometry", h5type, h5space);
  // Create an xy or xyz array
  T * xyz = new T[static_cast<size_t>(num_nodes) * dim];
  if (dim == 2) {
    for (size_t i = 0; i < num_nodes; ++i) {
      xyz[2 * i] = mesh.nodes_x[i];
      xyz[2 * i + 1] = mesh.nodes_y[i];
    }
  } else { // dim == 3
    for (size_t i = 0; i < num_nodes; ++i) {
      xyz[3 * i] = mesh.nodes_x[i];
      xyz[3 * i + 1] = mesh.nodes_y[i];
      xyz[3 * i + 2] = mesh.nodes_z[i];
    }
  }
  // Write HDF5 data set
  h5dataset.write(xyz, h5type, h5space);
  // If xyz is not null, delete it
  delete[] xyz;

} // write_xdmf_geometry

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void writeXDMFTopology(pugi::xml_node & xgrid, H5::Group & h5group,
                              std::string const & h5filename, std::string const & h5path,
                              MeshFile<T, I> const & mesh)
{
  Log::debug("Writing XDMF topology");
  // Create XDMF Topology node
  auto xtopo = xgrid.append_child("Topology");
  size_t const nelems = mesh.element_types.size();
  int8_t const element_type = mesh.element_types[0];

  struct IsHomogeneousFunctor {

    int8_t const element_type;

    PURE auto
    operator()(int8_t const a) const -> bool
    {
      return a == element_type;
    }
  } const homogeneous_functor{element_type};

  bool const is_homogeneous = std::all_of(mesh.element_types.cbegin(),
                                          mesh.element_types.cend(), homogeneous_functor);

  std::vector<I> topology;
  std::string topology_type;
  std::string dimensions;
  size_t nverts = 0;
  if (is_homogeneous) {
    if (element_type == static_cast<int8_t>(XDMFCellType::Triangle)) {
      topology_type = "Triangle";
      dimensions = std::to_string(nelems) + " 3";
      nverts = 3;
    } else if (element_type == static_cast<int8_t>(XDMFCellType::Quad)) {
      topology_type = "Quadrilateral";
      dimensions = std::to_string(nelems) + " 4";
      nverts = 4;
    } else if (element_type == static_cast<int8_t>(XDMFCellType::QuadraticTriangle)) {
      topology_type = "Triangle_6";
      dimensions = std::to_string(nelems) + " 6";
      nverts = 6;
    } else if (element_type == static_cast<int8_t>(XDMFCellType::QuadraticQuad)) {
      topology_type = "Quadrilateral_8";
      dimensions = std::to_string(nelems) + " 8";
      nverts = 8;
    } else {
      Log::error("Unsupported element type");
    }
  } else {
    topology_type = "Mixed";
    dimensions = std::to_string(nelems + mesh.element_conn.size());
    topology.resize(nelems + mesh.element_conn.size());
    // Create the topology array (type id + node ids)
    size_t topo_ctr = 0;
    for (size_t i = 0; i < nelems; ++i) {
      topology[topo_ctr] = static_cast<I>(static_cast<uint8_t>(mesh.element_types[i]));
      auto const offset = static_cast<size_t>(mesh.element_offsets[i]);
      auto const npts = static_cast<size_t>(mesh.element_offsets[i + 1]) - offset;
      for (size_t j = 0; j < npts; ++j) {
        topology[topo_ctr + j + 1] = mesh.element_conn[offset + j];
      }
      topo_ctr += npts + 1;
    }
  }
  xtopo.append_attribute("TopologyType") = topology_type.c_str();
  xtopo.append_attribute("NumberOfElements") = nelems;
  // Create XDMF DataItem node
  auto xdata = xtopo.append_child("DataItem");
  xdata.append_attribute("DataType") = "Int";
  xdata.append_attribute("Dimensions") = dimensions.c_str();
  if constexpr (sizeof(I) == 1) {
    xdata.append_attribute("Precision") = 1;
  } else if constexpr (sizeof(I) == 2) {
    xdata.append_attribute("Precision") = 2;
  } else if constexpr (sizeof(I) == 4) {
    xdata.append_attribute("Precision") = 4;
  } else if constexpr (sizeof(I) == 8) {
    xdata.append_attribute("Precision") = 8;
  }
  xdata.append_attribute("Format") = "HDF";
  std::string const h5topopath = h5filename + ":" + h5path + "/Topology";
  xdata.append_child(pugi::node_pcdata).set_value(h5topopath.c_str());

  // Create HDF5 data type
  H5::DataType h5type;
  // NOLINTBEGIN(cppcoreguidelines-slicing)
  if constexpr (sizeof(I) == 1) {
    h5type = H5::PredType::NATIVE_INT8;
  } else if constexpr (sizeof(I) == 2) {
    h5type = H5::PredType::NATIVE_INT16;
  } else if constexpr (sizeof(I) == 4) {
    h5type = H5::PredType::NATIVE_INT32;
  } else if constexpr (sizeof(I) == 8) {
    h5type = H5::PredType::NATIVE_INT64;
  } else {
    Log::error("Unsupported signed signed_integral type");
  }
  // NOLINTEND(cppcoreguidelines-slicing)

  if (topology_type == "Mixed") {
    // Create HDF5 data space
    auto dims = static_cast<hsize_t>(topology.size());
    H5::DataSpace const h5space(1, &dims);
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
    // Write HDF5 data set
    h5dataset.write(topology.data(), h5type, h5space);
  } else {
    // Create HDF5 data space
    hsize_t dims[2] = {static_cast<hsize_t>(nelems), nverts};
    H5::DataSpace const h5space(2, dims);
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
    // Write HDF5 data set
    h5dataset.write(mesh.element_conn.data(), h5type, h5space);
  }
} // writeXDMFTopology

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static void writeXDMFMaterials(pugi::xml_node & xgrid, H5::Group & h5group,
                               std::string const & h5filename, std::string const & h5path,
                               MeshFile<T, I> const & mesh,
                               std::vector<std::string> const & material_names)
{
  Log::debug("Writing XDMF materials");
  // Create material array
  size_t const nelems = mesh.element_types.size();
  std::vector<MaterialID> materials(nelems, -1);
  size_t const nmats = material_names.size();
  // Ensure that nmats fits in MaterialID without overflow
  if (nmats > std::numeric_limits<MaterialID>::max()) {
    Log::error("Number of materials exceeds MaterialID capacity");
  }
  for (size_t i = 0; i < nmats; ++i) {
    std::string const & mat_name = "Material_" + material_names[i];
    for (size_t j = 0; j < mesh.elset_names.size(); ++j) {
      if (mesh.elset_names[j] == mat_name) {
        auto const start = static_cast<size_t>(mesh.elset_offsets[j]);
        auto const end = static_cast<size_t>(mesh.elset_offsets[j + 1]);
        for (size_t k = start; k < end; ++k) {
          auto const elem = static_cast<size_t>(mesh.elset_ids[k]);
          if (materials[elem] != -1) {
            Log::error("Element " + std::to_string(elem) + " has multiple materials");
          }
          materials[elem] = static_cast<MaterialID>(i);
        } // for k
        break;
      } // if
    }   // for j
  }     // for i

  // Create HDF5 data space
  hsize_t dims = static_cast<hsize_t>(materials.size());
  H5::DataSpace const h5space(1, &dims);
  // Create HDF5 data type
  H5::DataType h5type;
  // NOLINTBEGIN(cppcoreguidelines-slicing)
  if constexpr (sizeof(MaterialID) == 1) {
    if constexpr (std::signed_integral<MaterialID>) {
      h5type = H5::PredType::NATIVE_INT8;
    } else if constexpr (std::signed_integral<MaterialID>) {
      h5type = H5::PredType::NATIVE_UINT8;
    }
  } else if constexpr (sizeof(MaterialID) == 2) {
    if constexpr (std::signed_integral<MaterialID>) {
      h5type = H5::PredType::NATIVE_INT16;
    } else if constexpr (std::signed_integral<MaterialID>) {
      h5type = H5::PredType::NATIVE_UINT16;
    }
  } else if constexpr (sizeof(MaterialID) == 4) {
    if constexpr (std::signed_integral<MaterialID>) {
      h5type = H5::PredType::NATIVE_INT32;
    } else if constexpr (std::signed_integral<MaterialID>) {
      h5type = H5::PredType::NATIVE_UINT32;
    }
  } else if constexpr (sizeof(MaterialID) == 8) {
    if constexpr (std::signed_integral<MaterialID>) {
      h5type = H5::PredType::NATIVE_INT64;
    } else if constexpr (std::signed_integral<MaterialID>) {
      h5type = H5::PredType::NATIVE_UINT64;
    }
  } else {
    Log::error("Unsupported MaterialID size");
  }
  // NOLINTEND(cppcoreguidelines-slicing)
  // Create HDF5 data set
  H5::DataSet h5dataset = h5group.createDataSet("Materials", h5type, h5space);
  // Write HDF5 data set
  h5dataset.write(materials.data(), h5type, h5space);

  // Create XDMF Materials node
  auto xmat = xgrid.append_child("Attribute");
  xmat.append_attribute("Name") = "Materials";
  xmat.append_attribute("Center") = "Cell";
  // Create XDMF DataItem node
  auto xdata = xmat.append_child("DataItem");
  if constexpr (std::signed_integral<MaterialID>) {
    xdata.append_attribute("DataType") = "Int";
  } else if constexpr (std::signed_integral<MaterialID>) {
    xdata.append_attribute("DataType") = "UInt";
  }
  xdata.append_attribute("Dimensions") = materials.size();
  if (sizeof(MaterialID) == 1) {
    xdata.append_attribute("Precision") = "1";
  } else if (sizeof(MaterialID) == 2) {
    xdata.append_attribute("Precision") = "2";
  } else if (sizeof(MaterialID) == 4) {
    xdata.append_attribute("Precision") = "4";
  } else if (sizeof(MaterialID) == 8) {
    xdata.append_attribute("Precision") = "8";
  } else {
    Log::error("Unsupported MaterialID size");
  }
  xdata.append_attribute("Format") = "HDF";
  std::string h5matpath = h5filename + ":" + h5path + "/Materials";
  xdata.append_child(pugi::node_pcdata).set_value(h5matpath.c_str());
} // write_xdmf_materials

// template <std::floating_point T, std::signed_integral I>
// requires (sizeof(T) == 4 || sizeof(T) == 8)
// static void write_xdmf_elsets(
//     pugi::xml_node & xgrid,
//     H5::Group & h5group,
//     std::string const & h5filename,
//     std::string const & h5path,
//     MeshFile<T, I> const & mesh)
//{
//     um2::Log::debug("Writing XDMF elsets");
//     for (size_t i = 0; i < mesh.elset_names.size(); ++i) {
//         std::string name = to_string(mesh.elset_names[i]);
//         size_t const start = static_cast<size_t>(mesh.elset_offsets[i    ]);
//         size_t const end   = static_cast<size_t>(mesh.elset_offsets[i + 1]);
//         // Create HDF5 data space
//         hsize_t dims = static_cast<hsize_t>(end - start);
//         H5::DataSpace h5space(1, &dims);
//         // Create HDF5 data type
//         H5::DataType h5type;
//         if constexpr (sizeof(I) == 1) {
//             h5type = H5::PredType::NATIVE_INT8;
//         } else if constexpr (sizeof(I) == 2) {
//             h5type = H5::PredType::NATIVE_INT16;
//         } else if constexpr (sizeof(I) == 4) {
//             h5type = H5::PredType::NATIVE_INT32;
//         } else if constexpr (sizeof(I) == 8) {
//             h5type = H5::PredType::NATIVE_INT64;
//         } else {
//             Log::error("Unsupported element index size");
//         }
//         // Create HDF5 data set
//         H5::DataSet h5dataset = h5group.createDataSet(name, h5type, h5space);
//         // Write HDF5 data set.
//         h5dataset.write(&mesh.elset_ids[start], h5type, h5space);
//
//         // Create XDMF Elset node
//         auto xelset = xgrid.append_child("Set");
//         xelset.append_attribute("Name") = name.c_str();
//         xelset.append_attribute("SetType") = "Cell";
//         // Create XDMF DataItem node
//         auto xdata = xelset.append_child("DataItem");
//         xdata.append_attribute("DataType") = "Int";
//         xdata.append_attribute("Dimensions") = end - start;
//         if constexpr (sizeof(I) == 1) {
//             xdata.append_attribute("Precision") = 1;
//         } else if constexpr (sizeof(I) == 2) {
//             xdata.append_attribute("Precision") = 2;
//         } else if constexpr (sizeof(I) == 4) {
//             xdata.append_attribute("Precision") = 4;
//         } else if constexpr (sizeof(I) == 8) {
//             xdata.append_attribute("Precision") = 8;
//         }
//         xdata.append_attribute("Format") = "HDF";
//         std::string h5elsetpath = h5filename + ":" + h5path + "/" + name;
//         xdata.append_child(pugi::node_pcdata).set_value(h5elsetpath.c_str());
//     }
// } // write_xdmf_elsets
//
template <std::floating_point T, std::signed_integral I>
static void
writeXDMFUniformGrid(pugi::xml_node & xdomain, H5::H5File & h5file,
                     std::string const & h5filename, MeshFile<T, I> const & mesh,
                     std::vector<std::string> const & material_names)
{
  Log::debug("Writing XDMF uniform grid");

  // Remove any leading slashes from the mesh name
  std::string const name_str = mesh.name;
  size_t name_end = name_str.find_last_of('/');
  std::string name;
  if (name_end == std::string::npos) {
    name = name_str;
  } else {
    name_end += 1;
    name = name_str.substr(name_end, name_str.size() - name_end);
  }

  // Grid
  pugi::xml_node xgrid = xdomain.append_child("Grid");
  xgrid.append_attribute("Name") = name.c_str();
  xgrid.append_attribute("GridType") = "Uniform";

  // h5
  std::string const h5groupname = name;
  std::string const h5grouppath = "/" + name;
  H5::Group h5group = h5file.createGroup(h5grouppath);

  writeXDMFGeometry(xgrid, h5group, h5filename, h5grouppath, mesh);
  writeXDMFTopology(xgrid, h5group, h5filename, h5grouppath, mesh);
  writeXDMFMaterials(xgrid, h5group, h5filename, h5grouppath, mesh, material_names);
  //    write_xdmf_elsets(xgrid, h5group, h5filename, h5grouppath, mesh);
} // write_xdmf_uniform_grid

template <std::floating_point T, std::signed_integral I>
void
writeXDMFFile(MeshFile<T, I> & mesh)
{

  // If format is Abaqus, convert to XDMF
  if (mesh.format == MeshFileFormat::Abaqus) {
    Log::info("Converting Abaqus mesh to XDMF");
    // Convert the element types
    std::transform(mesh.element_types.begin(), mesh.element_types.end(),
                   mesh.element_types.begin(), abaqus2xdmf);
    // Change the filepath ending if it is .inp
    auto const path_size = static_cast<size_t>(mesh.filepath.size());
    if (mesh.filepath.ends_with(".inp")) {
      mesh.filepath.data()[path_size - 3] = 'x';
      mesh.filepath.data()[path_size - 2] = 'd';
      mesh.filepath.data()[path_size - 1] = 'm';
      mesh.filepath.push_back('f');
    }
    // Change the format
    mesh.format = MeshFileFormat::XDMF;
  }

  Log::info("Writing XDMF file: " + mesh.filepath);

  // Check valid format
  if (mesh.format != MeshFileFormat::XDMF) {
    Log::error("Invalid mesh format: " +
               std::to_string(static_cast<int8_t>(mesh.format)));
    return;
  }

  // Setup HDF5 file
  // Get the h5 file name
  size_t const h5filepath_end = mesh.filepath.find_last_of('/') + 1;
  std::string const h5filename =
      mesh.filepath.substr(h5filepath_end, mesh.filepath.size() - 5 - h5filepath_end) +
      ".h5";
  std::string const h5filepath = mesh.filepath.substr(0, h5filepath_end);
  Log::debug("H5 filename: " + h5filename);
  H5::H5File h5file(h5filepath + h5filename, H5F_ACC_TRUNC);

  // Setup XML file
  pugi::xml_document xdoc;

  // XDMF root node
  pugi::xml_node xroot = xdoc.append_child("Xdmf");
  xroot.append_attribute("Version") = "3.0";

  // Domain node
  pugi::xml_node xdomain = xroot.append_child("Domain");

  // Get the material names from the mesh file elset names, in alphabetical order.
  std::vector<std::string> material_names;
  std::string const material = "Material";
  for (size_t i = 0; i < mesh.elset_names.size(); ++i) {
    size_t const name_len = mesh.elset_names[i].size();
    if (name_len >= 10 && mesh.elset_names[i].starts_with(material)) {
      material_names.push_back(mesh.elset_names[i].substr(9, name_len - 9));
    }
  }

  // If there are any materials, add an information node listing them
  if (!material_names.empty()) {
    pugi::xml_node xinfo = xdomain.append_child("Information");
    xinfo.append_attribute("Name") = "Materials";
    std::string materials;
    ;
    for (size_t i = 0; i < material_names.size(); i++) {
      materials += material_names[i];
      if (i + 1 < material_names.size()) {
        materials += ", ";
      }
    }
    xinfo.append_child(pugi::node_pcdata).set_value(materials.c_str());
  }

  // Add the mesh as a uniform grid
  writeXDMFUniformGrid(xdomain, h5file, h5filename, mesh, material_names);

  // Write the XML file
  xdoc.save_file(mesh.filepath.c_str(), "  ");

  // Close the HDF5 file
  h5file.close();

} // write_xdmf_file

//// -------------------------------------------------------------------------- //
//
// template <std::floating_point T, std::signed_integral I, std::floating_point V>
// static void add_nodes_to_mesh(
//        MeshFile<T, I> & mesh,
//        size_t const num_nodes,
//        size_t const num_dimensions,
//        H5::DataSet const & dataset,
//        H5::FloatType const & datatype,
//        bool const XYZ)
//{
//    V * data = new V [num_nodes * num_dimensions];
//    dataset.read(data, datatype);
//    size_t const num_nodes_old = mesh.nodes_x.size();
//    size_t const lnum_nodes = static_cast<size_t>(num_nodes);
//    // Add the nodes to the mesh
//    if (XYZ) {
//        mesh.nodes_x.insert(mesh.nodes_x.end(), lnum_nodes, -1);
//        mesh.nodes_y.insert(mesh.nodes_y.end(), lnum_nodes, -1);
//        mesh.nodes_z.insert(mesh.nodes_z.end(), lnum_nodes, -1);
//        for (size_t i = 0; i < lnum_nodes; ++i) {
//            mesh.nodes_x[num_nodes_old + i] = static_cast<T>(data[i * 3]);
//            mesh.nodes_y[num_nodes_old + i] = static_cast<T>(data[i * 3 + 1]);
//            mesh.nodes_z[num_nodes_old + i] = static_cast<T>(data[i * 3 + 2]);
//        }
//    } else { // XY
//        mesh.nodes_x.insert(mesh.nodes_x.end(), lnum_nodes, -1);
//        mesh.nodes_y.insert(mesh.nodes_y.end(), lnum_nodes, -1);
//        for (size_t i = 0; i < lnum_nodes; ++i) {
//            mesh.nodes_x[num_nodes_old + i] = static_cast<T>(data[i * 2]);
//            mesh.nodes_y[num_nodes_old + i] = static_cast<T>(data[i * 2 + 1]);
//        }
//    }
//    if (data) { delete[] data; }
//} // add_nodes_to_mesh
//
// template <std::floating_point T, std::signed_integral I>
// requires (sizeof(T) == 4 || sizeof(T) == 8)
// static void read_xdmf_geometry(
//    pugi::xml_node const & xgrid,
//    H5::H5File const & h5file,
//    std::string const & h5filename,
//    MeshFile<T, I> & mesh)
//{
//    Log::debug("Reading XDMF geometry");
//    pugi::xml_node xgeometry = xgrid.child("Geometry");
//    if (strcmp(xgeometry.name(), "Geometry") != 0) {
//        Log::error("XDMF geometry node not found");
//        return;
//    }
//    // Get the geometry type
//    std::string geometry_type = xgeometry.attribute("GeometryType").value();
//    if (geometry_type.compare("XYZ") != 0 &&
//        geometry_type.compare("XY") != 0) {
//        Log::error("XDMF geometry type not supported: " + geometry_type);
//        return;
//    }
//    // Get the DataItem node
//    pugi::xml_node xdataitem = xgeometry.child("DataItem");
//    if (strcmp(xdataitem.name(), "DataItem") != 0) {
//        Log::error("XDMF geometry DataItem node not found");
//        return;
//    }
//    // Get the data type
//    std::string data_type = xdataitem.attribute("DataType").value();
//    if (data_type.compare("Float") != 0) {
//        Log::error("XDMF geometry data type not supported: " + data_type);
//        return;
//    }
//    // Get the precision
//    std::string precision = xdataitem.attribute("Precision").value();
//    if (precision.compare("4") != 0 && precision.compare("8") != 0) {
//        Log::error("XDMF geometry precision not supported: " + precision);
//        return;
//    }
//    // Get the dimensions
//    std::string dimensions = xdataitem.attribute("Dimensions").value();
//    size_t const split = dimensions.find_last_of(" ");
//    size_t const num_nodes = std::stoul(dimensions.substr(0, split));
//    size_t const num_dimensions = std::stoul(dimensions.substr(split + 1));
//    if (geometry_type.compare("XYZ") == 0 && num_dimensions != 3) {
//        Log::error("XDMF geometry dimensions not supported: " + dimensions);
//        return;
//    }
//    if (geometry_type.compare("XY") == 0 && num_dimensions != 2) {
//        Log::error("XDMF geometry dimensions not supported: " + dimensions);
//        return;
//    }
//    // Get the format
//    std::string format = xdataitem.attribute("Format").value();
//    if (format.compare("HDF") != 0) {
//        Log::error("XDMF geometry format not supported: " + format);
//        return;
//    }
//    // Get the h5 dataset path
//    std::string h5dataset = xdataitem.child_value();
//    // Read the data
//    H5::DataSet dataset = h5file.openDataSet(h5dataset.substr(h5filename.size() + 1));
//    H5T_class_t type_class = dataset.getTypeClass();
//    UM2_ASSERT(type_class == H5T_FLOAT);
//    H5::FloatType datatype = dataset.getFloatType();
//    size_t const datatype_size = datatype.getsize_t();
//    UM2_ASSERT(datatype_size == std::stoul(precision));
//    H5::DataSpace dataspace = dataset.getSpace();
//    int rank = dataspace.getSimpleExtentNdims();
//    UM2_ASSERT(rank == 2);
//    hsize_t dims[2];
//    int ndims = dataspace.getSimpleExtentDims(dims, nullptr);
//    UM2_ASSERT(ndims == 2);
//    UM2_ASSERT(dims[0] == num_nodes);
//    UM2_ASSERT(dims[1] == num_dimensions);
//    if (datatype_size == 4) {
//        add_nodes_to_mesh<T, I, float>(mesh, num_nodes, num_dimensions,
//                                  dataset, datatype,
//                                  geometry_type.compare("XYZ") == 0);
//    } else if (datatype_size == 8) {
//        add_nodes_to_mesh<T, I, double>(mesh, num_nodes, num_dimensions,
//                                   dataset, datatype,
//                                   geometry_type.compare("XYZ") == 0);
//    }
//}
//
// template <std::floating_point T, std::signed_integral I, std::signed_integral V>
// static void add_elements_to_mesh(
//        size_t const num_elements,
//        std::string const & topology_type,
//        std::string const & dimensions,
//        MeshFile<T, I> & mesh,
//        H5::DataSet const & dataset,
//        H5::IntType const & datatype)
//{
//    size_t const prev_num_elements = mesh.element_types.size();
//    size_t const lnum_elements = static_cast<size_t>(num_elements);
//    if (prev_num_elements == 0) {
//        mesh.element_offsets.push_back(0);
//    }
//    I const prev_offset = mesh.element_offsets.back();
//    mesh.element_offsets.insert(mesh.element_offsets.end(), lnum_elements, -1);
//    if (topology_type.compare("Mixed") == 0) {
//        mesh.element_types.insert(mesh.element_types.end(), lnum_elements, -1);
//        // Expect dims to be one number
//        size_t const conn_length = sto<size_t>(dimensions);
//        V * data = new V[conn_length];
//        dataset.read(data, datatype);
//        // Add the elements to the mesh
//        size_t const prev_conn_size = mesh.element_conn.size();
//        size_t const num_conn_added = static_cast<size_t>(conn_length - num_elements);
//        mesh.element_conn.insert(mesh.element_conn.end(), num_conn_added, -1);
//        size_t offset = 0;
//        size_t position = 0;
//        for (size_t i = 0; i < lnum_elements; ++i) {
//            int8_t const element_type = static_cast<int8_t>(data[position]);
//            mesh.element_types[prev_num_elements + i] = element_type;
//            size_t const npoints = points_in_xdmf_cell(element_type);
//            for (size_t j = 0; j < npoints; ++j) {
//                mesh.element_conn[prev_conn_size + offset + j] =
//                    static_cast<I>(data[position + j + 1]);
//            }
//            offset += npoints;
//            position += npoints + 1;
//            mesh.element_offsets[1 + prev_num_elements + i] =
//                static_cast<I>(prev_offset + offset);
//        }
//        if (data) { delete[] data; }
//    } else {
//        size_t const split = dimensions.find_last_of(" ");
//        size_t const ncells = sto<size_t>(dimensions.substr(0, split));
//        size_t const nverts = sto<size_t>(dimensions.substr(split + 1));
//        UM2_ASSERT(ncells == lnum_elements);
//        V * data = new V[static_cast<size_t>(ncells * nverts)];
//        dataset.read(data, datatype);
//        // Add the elements to the mesh
//        size_t const prev_conn_size = mesh.element_conn.size();
//        mesh.element_conn.insert(mesh.element_conn.end(), ncells * nverts, -1);
//        for (size_t i = 0; i < ncells; ++i) {
//            mesh.element_offsets[1 + prev_num_elements + i] =
//                static_cast<I>((i + 1) * nverts) + prev_offset;
//            for (size_t j = 0; j < nverts; ++j) {
//                mesh.element_conn[prev_conn_size + i * nverts + j] =
//                    static_cast<I>(data[i * nverts + j]);
//            }
//        }
//        if (data) { delete[] data; }
//        int8_t element_type = -1;
//        if (topology_type.compare("Triangle") == 0) {
//            element_type = static_cast<int8_t>(XDMFCellType::TRIANGLE);
//        } else if (topology_type.compare("Quadrilateral") == 0) {
//            element_type = static_cast<int8_t>(XDMFCellType::QUAD);
//        } else if (topology_type.compare("Triangle_6") == 0) {
//            element_type = static_cast<int8_t>(XDMFCellType::QUADRATIC_TRIANGLE);
//        } else if (topology_type.compare("Quadrilateral_8") == 0) {
//            element_type = static_cast<int8_t>(XDMFCellType::QUADRATIC_QUAD);
//        } else {
//            Log::error("Unsupported element type");
//        }
//        mesh.element_types.insert(mesh.element_types.end(), ncells, element_type);
//    }
//}
//
// template <std::floating_point T, std::signed_integral I>
// requires (sizeof(T) == 4 || sizeof(T) == 8)
// static void read_xdmf_topology(
//    pugi::xml_node const & xgrid,
//    H5::H5File const & h5file,
//    std::string const & h5filename,
//    MeshFile<T, I> & mesh)
//{
//    Log::debug("Reading XDMF topology");
//    pugi::xml_node xtopology = xgrid.child("Topology");
//    if (strcmp(xtopology.name(), "Topology") != 0) {
//        Log::error("XDMF topology node not found");
//        return;
//    }
//    // Get the topology type
//    std::string topology_type = xtopology.attribute("TopologyType").value();
//    // Get the number of elements
//    size_t const num_elements =
//    std::stoul(xtopology.attribute("NumberOfElements").value());
//    // Get the DataItem node
//    pugi::xml_node xdataitem = xtopology.child("DataItem");
//    if (strcmp(xdataitem.name(), "DataItem") != 0) {
//        Log::error("XDMF topology DataItem node not found");
//        return;
//    }
//    // Get the data type
//    std::string data_type = xdataitem.attribute("DataType").value();
//    if (data_type.compare("Int") != 0) {
//        Log::error("XDMF topology data type not supported: " + data_type);
//        return;
//    }
//    // Get the precision
//    std::string precision = xdataitem.attribute("Precision").value();
//    if (precision.compare("1") != 0 && precision.compare("2") != 0 &&
//        precision.compare("4") != 0 && precision.compare("8") != 0) {
//        Log::error("XDMF topology precision not supported: " + precision);
//        return;
//    }
//    // Get the format
//    std::string format = xdataitem.attribute("Format").value();
//    if (format.compare("HDF") != 0) {
//        Log::error("XDMF geometry format not supported: " + format);
//        return;
//    }
//    // Get the h5 dataset path
//    std::string h5dataset = xdataitem.child_value();
//    // Read the data
//    H5::DataSet dataset = h5file.openDataSet(h5dataset.substr(h5filename.size() + 1));
//    H5T_class_t type_class = dataset.getTypeClass();
//    UM2_ASSERT(type_class == H5T_INTEGER);
//    H5::IntType datatype = dataset.getIntType();
//    size_t const datatype_size = datatype.getsize_t();
//    UM2_ASSERT(datatype_size == std::stoul(precision));
//    H5::DataSpace dataspace = dataset.getSpace();
//    int rank = dataspace.getSimpleExtentNdims();
//    if (topology_type.compare("Mixed") == 0) {
//        UM2_ASSERT(rank == 1);
//        hsize_t dims[1];
//        int ndims = dataspace.getSimpleExtentDims(dims, nullptr);
//        UM2_ASSERT(ndims == 1);
//    } else {
//        UM2_ASSERT(rank == 2);
//        hsize_t dims[2];
//        int ndims = dataspace.getSimpleExtentDims(dims, nullptr);
//        UM2_ASSERT(ndims == 2);
//    }
//    // Get the dimensions
//    std::string dimensions = xdataitem.attribute("Dimensions").value();
//    if (datatype_size == 1) {
//        add_elements_to_mesh<T, I, int8_t>(num_elements, topology_type,
//                                           dimensions, mesh, dataset, datatype);
//    } else if (datatype_size == 2) {
//        add_elements_to_mesh<T, I, int16_t>(num_elements, topology_type,
//                                            dimensions, mesh, dataset, datatype);
//    } else if (datatype_size == 4) {
//        add_elements_to_mesh<T, I, int32_t>(num_elements, topology_type,
//                                            dimensions, mesh, dataset, datatype);
//    } else if (datatype_size == 8) {
//        add_elements_to_mesh<T, I, int64_t>(num_elements, topology_type,
//                                            dimensions, mesh, dataset, datatype);
//    } else {
//        Log::error("Unsupported data type size");
//    }
//}
//
////template <std::floating_point T, std::signed_integral I>
////requires (sizeof(T) == 4 || sizeof(T) == 8)
////static void read_xdmf_materials(
////    pugi::xml_node const & xgrid,
////    H5::H5File const & h5file,
////    std::string const & h5filename,
////    std::vector<std::string> const & material_names,
////    MeshFile<T, I> & mesh)
////{
////    Log::debug("Reading XDMF materials");
////    // Loop over all nodes to find the materials
////    for (pugi::xml_node xmaterial = xgrid.first_child();
////         xmaterial; xmaterial = xmaterial.next_sibling()) {
////        if (strcmp(xmaterial.name(), "Attribute") != 0) {
////            continue;
////        }
////        // Get the name
////        std::string name = xmaterial.attribute("Name").value();
////        if (name.compare("Materials") != 0) {
////            continue;
////        }
////        // Get the center
////        std::string center = xmaterial.attribute("Center").value();
////        if (center.compare("Cell") != 0) {
////            Log::error("XDMF material only supports Center=Cell");
////            return;
////        }
////        // Get the DataItem node
////        pugi::xml_node xdataitem = xmaterial.child("DataItem");
////        if (strcmp(xdataitem.name(), "DataItem") != 0) {
////            Log::error("XDMF material DataItem node not found");
////            return;
////        }
////        // Get the data type
////        std::string data_type = xdataitem.attribute("DataType").value();
////        if (data_type.compare("UInt") != 0 && data_type.compare("Int") != 0) {
////            Log::error("XDMF material data type not supported: " + data_type);
////            return;
////        }
////        // Get the precision
////        std::string precision = xdataitem.attribute("Precision").value();
////        if (precision.compare("1") != 0) {
////            Log::error("XDMF material precision not supported: " + precision);
////            return;
////        }
////        // Get the format
////        std::string format = xdataitem.attribute("Format").value();
////        if (format.compare("HDF") != 0) {
////            Log::error("XDMF material format not supported: " + format);
////            return;
////        }
////        // Get the h5 dataset path
////        std::string h5dataset = xdataitem.child_value();
////        // Read the data
////        H5::DataSet dataset = h5file.openDataSet(h5dataset.substr(h5filename.size() +
/// 1)); /        H5T_class_t type_class = dataset.getTypeClass(); / UM2_ASSERT(type_class
///== H5T_INTEGER); /        H5::IntType datatype = dataset.getIntType(); /        size_t
/// const datatype_size = datatype.getsize_t(); /        UM2_ASSERT(datatype_size ==
/// std::stoul(precision)); /        H5::DataSpace dataspace = dataset.getSpace(); / int
/// rank = dataspace.getSimpleExtentNdims(); /        UM2_ASSERT(rank == 1); / hsize_t
/// dims[1]; /        int ndims = dataspace.getSimpleExtentDims(dims, nullptr); /
/// UM2_ASSERT(ndims == 1); /        // Get the dimensions /        std::string dimensions
///= xdataitem.attribute("Dimensions").value(); /        UM2_ASSERT(datatype_size == 1);
////        // Expect dims to be one number
////        size_t const mats_length = dims[0];
////        std::vector<int8_t> data(mats_length);
////        dataset.read(data.data(), datatype);
////        for (size_t i = 0; i < material_names.size(); ++i) {
////            std::vector<I> & mat = mesh.elsets["Material:_" + material_names[i]];
////            long const ctr = std::count(data.cbegin(), data.cend(),
////                                          static_cast<int8_t>(i));
////            mat.reserve(static_cast<size_t>(ctr));
////            for (size_t j = 0; j < mats_length; ++j) {
////                if (data[j] == static_cast<int8_t>(i)) {
////                    mat.push_back(static_cast<I>(j));
////                }
////            }
////        }
////        return;
////    }
////}
//
// template <std::floating_point T, std::signed_integral I, std::signed_integral V>
// static void add_elset_to_mesh(
//        MeshFile<T, I> & mesh,
//        size_t const num_elements,
//        H5::DataSet const & dataset,
//        H5::IntType const & datatype)
//{
//    V * data = new V[num_elements];
//    dataset.read(data, datatype);
//    size_t const lnum_elements = static_cast<size_t>(num_elements);
//    size_t const offset = static_cast<size_t>(mesh.elset_offsets.back());
//    I const last_offset = mesh.elset_offsets.back();
//    mesh.elset_offsets.push_back(last_offset + static_cast<I>(num_elements));
//    mesh.elset_ids.insert(mesh.elset_ids.end(),
//                          lnum_elements, -1);
//    for (size_t i = 0; i < lnum_elements; ++i) {
//        mesh.elset_ids[i + offset] = static_cast<I>(data[i]);
//    }
//    if (data) { delete[] data; }
//}
//
// template <std::floating_point T, std::signed_integral I>
// requires (sizeof(T) == 4 || sizeof(T) == 8)
// static void read_xdmf_elsets(
//    pugi::xml_node const & xgrid,
//    H5::H5File const & h5file,
//    std::string const & h5filename,
//    MeshFile<T, I> & mesh)
//{
//    Log::debug("Reading XDMF elsets");
//    // Loop over all nodes to find the elsets
//    for (pugi::xml_node xelset = xgrid.first_child();
//         xelset; xelset = xelset.next_sibling()) {
//        if (strcmp(xelset.name(), "Set") != 0) {
//            continue;
//        }
//        // Get the SetType
//        std::string set_type = xelset.attribute("SetType").value();
//        if (set_type.compare("Cell") != 0) {
//            Log::error("XDMF elset only supports SetType=Cell");
//            return;
//        }
//        // Get the name
//        std::string name = xelset.attribute("Name").value();
//        if (name.empty()) {
//            Log::error("XDMF elset name not found");
//            return;
//        }
//        // Get the DataItem node
//        pugi::xml_node xdataitem = xelset.child("DataItem");
//        if (strcmp(xdataitem.name(), "DataItem") != 0) {
//            Log::error("XDMF elset DataItem node not found");
//            return;
//        }
//        // Get the data type
//        std::string data_type = xdataitem.attribute("DataType").value();
//        if (data_type.compare("Int") != 0) {
//            Log::error("XDMF elset data type not supported: " + data_type);
//            return;
//        }
//        // Get the precision
//        std::string precision = xdataitem.attribute("Precision").value();
//        if (precision.compare("1") != 0 && precision.compare("2") != 0 &&
//            precision.compare("4") != 0 && precision.compare("8") != 0) {
//            Log::error("XDMF elset precision not supported: " + precision);
//            return;
//        }
//        // Get the format
//        std::string format = xdataitem.attribute("Format").value();
//        if (format.compare("HDF") != 0) {
//            Log::error("XDMF elset format not supported: " + format);
//            return;
//        }
//        // Get the h5 dataset path
//        std::string h5dataset = xdataitem.child_value();
//        // Read the data
//        H5::DataSet dataset = h5file.openDataSet(h5dataset.substr(h5filename.size() +
//        1)); H5T_class_t type_class = dataset.getTypeClass(); UM2_ASSERT(type_class ==
//        H5T_INTEGER); H5::IntType datatype = dataset.getIntType(); size_t const
//        datatype_size = datatype.getsize_t(); UM2_ASSERT(datatype_size ==
//        std::stoul(precision)); H5::DataSpace dataspace = dataset.getSpace(); int rank =
//        dataspace.getSimpleExtentNdims(); UM2_ASSERT(rank == 1); hsize_t dims[1]; int
//        ndims = dataspace.getSimpleExtentDims(dims, nullptr); UM2_ASSERT(ndims == 1);
//        // Get the dimensions
//        std::string dimensions = xdataitem.attribute("Dimensions").value();
//        size_t const num_elements = dims[0];
//        UM2_ASSERT(num_elements == std::stoul(dimensions));
//        mesh.elset_names.push_back(name);
//        if (mesh.elset_offsets.empty()) {
//            mesh.elset_offsets.push_back(0);
//        }
//        if (datatype_size == 1) {
//            add_elset_to_mesh<T, I, int8_t>(mesh, num_elements, dataset, datatype);
//        } else if (datatype_size == 2) {
//            add_elset_to_mesh<T, I, int16_t>(mesh, num_elements, dataset, datatype);
//        } else if (datatype_size == 4) {
//            add_elset_to_mesh<T, I, int32_t>(mesh, num_elements, dataset, datatype);
//        } else if (datatype_size == 8) {
//            add_elset_to_mesh<T, I, int64_t>(mesh, num_elements, dataset, datatype);
//        }
//    }
//}
//
// template <std::floating_point T, std::signed_integral I>
// requires (sizeof(T) == 4 || sizeof(T) == 8)
// void read_xdmf_uniform_grid(
//    pugi::xml_node const & xgrid,
//    H5::H5File const & h5file,
//    std::string const & h5filename,
////    std::vector<std::string> const & material_names,
//    MeshFile<T, I> & mesh)
//{
//    read_xdmf_geometry(xgrid, h5file, h5filename, mesh);
//    read_xdmf_topology(xgrid, h5file, h5filename, mesh);
//    // Reading materials is redundant with elsets
////    read_xdmf_materials(xgrid, h5file, h5filename, material_names, mesh);
//    read_xdmf_elsets(xgrid, h5file, h5filename, mesh);
//}
//
// template <std::floating_point T, std::signed_integral I>
// void read_xdmf_file(std::string const & filename, MeshFile<T, I> & mesh)
//{
//    Log::info("Reading XDMF mesh file: " + filename);
//
//    // Open the XDMF file
//    std::ifstream file(filename);
//    if (!file.is_open()) {
//        Log::error("Could not open file: " + filename);
//        return;
//    }
//    // Open HDF5 file
//    size_t const h5filepath_end = filename.find_last_of("/") + 1;
//    std::string h5filename = filename.substr(
//            h5filepath_end,
//            filename.size() - 4 - h5filepath_end) + "h5";
//    std::string h5filepath = filename.substr(0, h5filepath_end);
//    um2::Log::debug("H5 filename: " + h5filename);
//    H5::H5File h5file(h5filepath + h5filename, H5F_ACC_RDONLY);
//
//    // Set filepath and format
//    mesh.filepath = filename;
//    mesh.format = MeshFileFormat::XDMF;
//
//    // Setup XML file
//    pugi::xml_document xdoc;
//    pugi::xml_parse_result result = xdoc.load_file(filename.c_str());
//    if (!result) {
//        Log::error("XDMF XML parse error: " + std::string(result.description()) +
//                ", character pos= " + std::to_string(result.offset));
//    }
//    pugi::xml_node xroot = xdoc.child("Xdmf");
//    if (strcmp("Xdmf", xroot.name()) != 0) {
//        Log::error("XDMF XML root node is not Xdmf");
//        return;
//    }
//    pugi::xml_node xdomain = xroot.child("Domain");
//    if (strcmp("Domain", xdomain.name()) != 0) {
//        Log::error("XDMF XML domain node is not Domain");
//        return;
//    }
//    std::vector<std::string> material_names;
//    pugi::xml_node xinfo = xdomain.child("Information");
//    if (strcmp("Information", xinfo.name()) == 0) {
//        // Get the "Name" attribute
//        pugi::xml_attribute xname = xinfo.attribute("Name");
//        if (strcmp("Materials", xname.value()) == 0) {
//            // Get the material names
//            std::string materials = xinfo.child_value();
//            std::stringstream ss(materials);
//            std::string material;
//            while (std::getline(ss, material, ',')) {
//                if (material[0] == ' ') {
//                    material_names.push_back(material.substr(1));
//                } else {
//                    material_names.push_back(material);
//                }
//            }
//        }
//    }
//    pugi::xml_node xgrid = xdomain.child("Grid");
//    if (strcmp("Grid", xgrid.name()) != 0) {
//        Log::error("XDMF XML grid node is not Grid");
//        return;
//    }
//    std::string grid_name = xgrid.attribute("Name").value();
//    mesh.name = grid_name;
//    if (strcmp("Uniform", xgrid.attribute("GridType").value()) == 0) {
//        read_xdmf_uniform_grid(xgrid, h5file, h5filename, mesh);
//    } else if (strcmp("Tree", xgrid.attribute("GridType").value()) == 0) {
//        Log::error("XDMF XML Tree is not supported");
//    } else {
//        Log::error("XDMF XML grid type is not Uniform or Tree");
//    }
//}

} // namespace um2
