#pragma once

#include <um2/mesh/PolytopeSoup.hpp>


#include <iostream>

//#include <um2/stdlib/sto.hpp>
//
//#include <cstring>   // strcmp
//#include <sstream>   // std::stringstream
//#include <string>
//#include <vector>

// External dependencies
#include <H5Cpp.h>     // H5::H5File, H5::DataSet, H5::DataSpace, H5::DataType
#include <pugixml.hpp> // pugi::xml_document, pugi::xml_node
                       // H5::Group, H5::PredType, H5::hsize_t

// Turn off useless cast warnings, since the casts are only useless for certain
// CMake configurations.
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wuseless-cast"
#endif
namespace um2
{

template <typename T>
static inline auto
getH5DataType() -> H5::PredType
{
  // NOLINTNEXTLINE(bugprone-branch-clone) justification: Need a default case
  if constexpr (std::same_as<T, float>) {
    return H5::PredType::NATIVE_FLOAT;
  } else if constexpr (std::same_as<T, double>) {
    return H5::PredType::NATIVE_DOUBLE;
  } else if constexpr (std::same_as<T, int8_t>) {
    return H5::PredType::NATIVE_INT8;
  } else if constexpr (std::same_as<T, int16_t>) {
    return H5::PredType::NATIVE_INT16;
  } else if constexpr (std::same_as<T, int32_t>) {
    return H5::PredType::NATIVE_INT32;
  } else if constexpr (std::same_as<T, int64_t>) {
    return H5::PredType::NATIVE_INT64;
  } else if constexpr (std::same_as<T, uint8_t>) {
    return H5::PredType::NATIVE_UINT8;
  } else if constexpr (std::same_as<T, uint16_t>) {
    return H5::PredType::NATIVE_UINT16;
  } else if constexpr (std::same_as<T, uint32_t>) {
    return H5::PredType::NATIVE_UINT32;
  } else if constexpr (std::same_as<T, uint64_t>) {
    return H5::PredType::NATIVE_UINT64;
  } else {
    static_assert(always_false<T>, "Unsupported type");
    return H5::PredType::NATIVE_FLOAT;
  }
}

////==============================================================================
//// writeXDMFGeometry
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
//static void writeXDMFGeometry(pugi::xml_node & xgrid, H5::Group & h5group,
//                              std::string const & h5filename, std::string const & h5path,
//                              PolytopeSoup<T, I> const & mesh)
//{
//  LOG_DEBUG("Writing XDMF geometry");
//  size_t const num_verts = mesh.vertices.size();
//  bool const is_2d =
//      std::count_if(mesh.vertices.cbegin(), mesh.vertices.cend(), [](auto const & v) {
//        return um2::abs(v[2]) < eps_distance<T>;
//      }) == static_cast<int64_t>(num_verts);
//  size_t const dim = is_2d ? 2 : 3;
//  // Create XDMF Geometry node
//  auto xgeom = xgrid.append_child("Geometry");
//  if (dim == 3) {
//    xgeom.append_attribute("GeometryType") = "XYZ";
//  } else { // (dim == 2)
//    xgeom.append_attribute("GeometryType") = "XY";
//  }
//
//  // Create XDMF DataItem node
//  auto xdata = xgeom.append_child("DataItem");
//  xdata.append_attribute("DataType") = "Float";
//  xdata.append_attribute("Dimensions") =
//      (std::to_string(num_verts) + " " + std::to_string(dim)).c_str();
//  xdata.append_attribute("Precision") = sizeof(T);
//  xdata.append_attribute("Format") = "HDF";
//  std::string const h5geompath = h5filename + ":" + h5path + "/Geometry";
//  xdata.append_child(pugi::node_pcdata).set_value(h5geompath.c_str());
//
//  // Create HDF5 data space
//  hsize_t dims[2] = {static_cast<hsize_t>(num_verts), dim};
//  H5::DataSpace const h5space(2, dims);
//  // Create HDF5 data type
//  H5::DataType const h5type = getH5DataType<T>();
//  // Create HDF5 data set
//  H5::DataSet const h5dataset = h5group.createDataSet("Geometry", h5type, h5space);
//  // Create an xy or xyz array
//  // cppcheck-suppress constStatement; justification: cppcheck is scared *const
//  T * const xyz = new T[num_verts * dim];
//  if (dim == 2) {
//    for (size_t i = 0; i < num_verts; ++i) {
//      xyz[2 * i] = mesh.vertices[i][0];
//      xyz[2 * i + 1] = mesh.vertices[i][1];
//    }
//  } else { // dim == 3
//    for (size_t i = 0; i < num_verts; ++i) {
//      xyz[3 * i] = mesh.vertices[i][0];
//      xyz[3 * i + 1] = mesh.vertices[i][1];
//      xyz[3 * i + 2] = mesh.vertices[i][2];
//    }
//  }
//  // Write HDF5 data set
//  h5dataset.write(xyz, h5type, h5space);
//  // If xyz is not null, delete it
//  delete[] xyz;
//
//} // writeXDMFgeometry
//
////==============================================================================
//// writeXDMFTopology
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
//static void writeXDMFTopology(pugi::xml_node & xgrid, H5::Group & h5group,
//                              std::string const & h5filename, std::string const & h5path,
//                              PolytopeSoup<T, I> const & mesh)
//{
//  LOG_DEBUG("Writing XDMF topology");
//  // Create XDMF Topology node
//  auto xtopo = xgrid.append_child("Topology");
//  size_t const ncells = mesh.numCells();
//
//  std::vector<I> topology;
//  std::string topology_type;
//  std::string dimensions;
//  size_t nverts = 0;
//  bool ishomogeneous = true;
//  MeshType const mesh_type = mesh.getMeshType();
//  if (mesh_type == MeshType::Tri) {
//    topology_type = "Triangle";
//    dimensions = std::to_string(ncells) + " 3";
//    nverts = 3;
//  } else if (mesh_type == MeshType::Quad) {
//    topology_type = "Quadrilateral";
//    dimensions = std::to_string(ncells) + " 4";
//    nverts = 4;
//  } else if (mesh_type == MeshType::QuadraticTri) {
//    topology_type = "Triangle_6";
//    dimensions = std::to_string(ncells) + " 6";
//    nverts = 6;
//  } else if (mesh_type == MeshType::QuadraticQuad) {
//    topology_type = "Quadrilateral_8";
//    dimensions = std::to_string(ncells) + " 8";
//    nverts = 8;
//  } else if (mesh_type == MeshType::TriQuad || mesh_type == MeshType::QuadraticTriQuad) {
//    topology_type = "Mixed";
//    ishomogeneous = false;
//    dimensions = std::to_string(ncells + mesh.element_conn.size());
//    topology.resize(ncells + mesh.element_conn.size());
//    // Create the topology array (type id + node ids)
//    size_t topo_ctr = 0;
//    for (size_t i = 0; i < ncells; ++i) {
//      int8_t const topo_type = meshTypeToXDMFElemType(mesh.element_types[i]);
//      if (topo_type == -1) {
//        Log::error("Unsupported mesh type");
//      }
//      topology[topo_ctr] = static_cast<I>(static_cast<unsigned int>(topo_type));
//      auto const offset = static_cast<size_t>(mesh.element_offsets[i]);
//      auto const npts =
//          static_cast<size_t>(mesh.element_offsets[i + 1] - mesh.element_offsets[i]);
//      for (size_t j = 0; j < npts; ++j) {
//        topology[topo_ctr + j + 1] = mesh.element_conn[offset + j];
//      }
//      topo_ctr += npts + 1;
//    }
//  } else {
//    Log::error("Unsupported mesh type");
//  }
//  xtopo.append_attribute("TopologyType") = topology_type.c_str();
//  xtopo.append_attribute("NumberOfElements") = ncells;
//  // Create XDMF DataItem node
//  auto xdata = xtopo.append_child("DataItem");
//  xdata.append_attribute("DataType") = "Int";
//  xdata.append_attribute("Dimensions") = dimensions.c_str();
//  xdata.append_attribute("Precision") = sizeof(I);
//  xdata.append_attribute("Format") = "HDF";
//  std::string const h5topopath = h5filename + ":" + h5path + "/Topology";
//  xdata.append_child(pugi::node_pcdata).set_value(h5topopath.c_str());
//
//  // Create HDF5 data type
//  H5::DataType const h5type = getH5DataType<I>();
//  if (ishomogeneous) {
//    // Create HDF5 data space
//    hsize_t dims[2] = {static_cast<hsize_t>(ncells), nverts};
//    H5::DataSpace const h5space(2, dims);
//    // Create HDF5 data set
//    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
//    // Write HDF5 data set
//    h5dataset.write(mesh.element_conn.data(), h5type, h5space);
//  } else {
//    // Create HDF5 data space
//    auto const dims = static_cast<hsize_t>(topology.size());
//    H5::DataSpace const h5space(1, &dims);
//    // Create HDF5 data set
//    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
//    // Write HDF5 data set
//    h5dataset.write(topology.data(), h5type, h5space);
//  }
//} // writeXDMFTopology
//
////==============================================================================
//// writeXDMFMaterials
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
//static void writeXDMFMaterials(pugi::xml_node & xgrid, H5::Group & h5group,
//                               std::string const & h5filename, std::string const & h5path,
//                               PolytopeSoup<T, I> const & mesh,
//                               std::vector<std::string> const & material_names)
//{
//  LOG_DEBUG("Writing XDMF materials");
//  // Create material array
//  size_t const ncells = mesh.numCells();
//  std::vector<MaterialID> materials(ncells, -1);
//  size_t const nmats = material_names.size();
//  // Ensure that nmats fits in MaterialID without overflow
//  if (nmats > std::numeric_limits<MaterialID>::max()) {
//    Log::error("Number of materials exceeds MaterialID capacity");
//  }
//  for (size_t i = 0; i < nmats; ++i) {
//    std::string const & mat_name = "Material_" + material_names[i];
//    for (size_t j = 0; j < mesh.elset_names.size(); ++j) {
//      if (mesh.elset_names[j] == mat_name) {
//        auto const start = static_cast<size_t>(mesh.elset_offsets[j]);
//        auto const end = static_cast<size_t>(mesh.elset_offsets[j + 1]);
//        for (size_t k = start; k < end; ++k) {
//          auto const elem = static_cast<size_t>(mesh.elset_ids[k]);
//          if (materials[elem] != -1) {
//            Log::error("Element " + toString(elem) + " has multiple materials");
//          }
//          materials[elem] = static_cast<MaterialID>(i);
//        } // for k
//        break;
//      } // if
//    }   // for j
//  }     // for i
//
//  // Create HDF5 data space
//  auto const dims = static_cast<hsize_t>(materials.size());
//  H5::DataSpace const h5space(1, &dims);
//  // Create HDF5 data type
//  static_assert(std::signed_integral<MaterialID>);
//  H5::DataType const h5type = getH5DataType<MaterialID>();
//  // Create HDF5 data set
//  H5::DataSet const h5dataset = h5group.createDataSet("Materials", h5type, h5space);
//  // Write HDF5 data set
//  h5dataset.write(materials.data(), h5type, h5space);
//
//  // Create XDMF Materials node
//  auto xmat = xgrid.append_child("Attribute");
//  xmat.append_attribute("Name") = "Materials";
//  xmat.append_attribute("Center") = "Cell";
//  // Create XDMF DataItem node
//  auto xdata = xmat.append_child("DataItem");
//  xdata.append_attribute("DataType") = "Int";
//  xdata.append_attribute("Dimensions") = materials.size();
//  xdata.append_attribute("Precision") = sizeof(MaterialID);
//  xdata.append_attribute("Format") = "HDF";
//  std::string const h5matpath = h5filename + ":" + h5path + "/Materials";
//  xdata.append_child(pugi::node_pcdata).set_value(h5matpath.c_str());
//} // writeXDMFMaterials
//
////==============================================================================
//// writeXDMFElsets
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
//static void writeXDMFElsets(pugi::xml_node & xgrid, H5::Group & h5group,
//                            std::string const & h5filename, std::string const & h5path,
//                            PolytopeSoup<T, I> const & mesh)
//{
//  LOG_DEBUG("Writing XDMF elsets");
//  for (size_t i = 0; i < mesh.elset_names.size(); ++i) {
//    std::string const name = mesh.elset_names[i];
//    auto const start = static_cast<size_t>(mesh.elset_offsets[i]);
//    auto const end = static_cast<size_t>(mesh.elset_offsets[i + 1]);
//    // Create HDF5 data space
//    auto dims = static_cast<hsize_t>(end - start);
//    H5::DataSpace const h5space(1, &dims);
//    // Create HDF5 data type
//    H5::DataType const h5type = getH5DataType<I>();
//    // Create HDF5 data set
//    H5::DataSet const h5dataset = h5group.createDataSet(name, h5type, h5space);
//    // Write HDF5 data set.
//    h5dataset.write(&mesh.elset_ids[start], h5type, h5space);
//
//    // Create XDMF Elset node
//    auto xelset = xgrid.append_child("Set");
//    xelset.append_attribute("Name") = name.c_str();
//    xelset.append_attribute("SetType") = "Cell";
//    // Create XDMF DataItem node
//    auto xdata = xelset.append_child("DataItem");
//    xdata.append_attribute("DataType") = "Int";
//    xdata.append_attribute("Dimensions") = end - start;
//    xdata.append_attribute("Precision") = sizeof(I);
//    xdata.append_attribute("Format") = "HDF";
//    std::string h5elsetpath = h5filename;
//    h5elsetpath += ':';
//    h5elsetpath += h5path;
//    h5elsetpath += '/';
//    h5elsetpath += name;
//    xdata.append_child(pugi::node_pcdata).set_value(h5elsetpath.c_str());
//  }
//} // writeXDMFelsets
//
//==============================================================================
// writeXDMFUniformGrid
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
writeXDMFUniformGrid(pugi::xml_node & xdomain, H5::H5File & /*h5file*/,
                     String const & /*h5filename*/, String const & /*h5path*/,
                     PolytopeSoup<T, I> const & /*mesh*/,
                     String const & name,
                     Vector<String> const & /*material_names*/)
{
  LOG_DEBUG("Writing XDMF uniform grid");

  // Grid
  pugi::xml_node xgrid = xdomain.append_child("Grid");
  xgrid.append_attribute("Name") = name.c_str();
  xgrid.append_attribute("GridType") = "Uniform";

  // h5
//  String const h5grouppath = h5path + "/" + name;
//  H5::Group const h5group = h5file.createGroup(h5grouppath.c_str());

//  writeXDMFGeometry(xgrid, h5group, h5filename, h5grouppath, mesh);
//  writeXDMFTopology(xgrid, h5group, h5filename, h5grouppath, mesh);
//  writeXDMFMaterials(xgrid, h5group, h5filename, h5grouppath, mesh, material_names);
//  writeXDMFElsets(xgrid, h5group, h5filename, h5grouppath, mesh);
} // writeXDMFUniformGrid

//==============================================================================
// writeXDMFFile
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
writeXDMFFile(String const & filepath, PolytopeSoup<T, I> & soup, String const & name = "UM2")
{

  Log::info("Writing XDMF file: " + filepath);

  // Setup HDF5 file
  // Get the h5 file name
  Size const h5filepath_end = filepath.find_last_of('/') + 1;
  String const h5filename =
      filepath.substr(h5filepath_end, filepath.size() - 5 - h5filepath_end) +
      ".h5";
  String const h5filepath = filepath.substr(0, h5filepath_end);
  LOG_DEBUG("H5 filename: " + h5filename);
  H5::H5File h5file((h5filepath + h5filename).c_str(), H5F_ACC_TRUNC);

  // Setup XML file
  pugi::xml_document xdoc;

  // XDMF root node
  pugi::xml_node xroot = xdoc.append_child("Xdmf");
  xroot.append_attribute("Version") = "3.0";

  // Domain node
  pugi::xml_node xdomain = xroot.append_child("Domain");

  // Get the material names from elset names, in alphabetical order.
  Vector<String> material_names;
  soup.getMaterialNames(material_names);
  std::sort(material_names.begin(), material_names.end());
  // Remove "Material_" from the beginning of each material name
  for (auto & mat_name : material_names) {
    std::cerr << mat_name.c_str() << std::endl;
    std::cerr << mat_name.size() << std::endl;
    String const short_name = mat_name.substr(9, mat_name.size() - 9);
    std::cerr << short_name.c_str() << std::endl;
    std::cerr << short_name.size() << std::endl;
    mat_name = short_name; 
  }

  // If there are any materials, add an information node listing them
  if (!material_names.empty()) {
    pugi::xml_node xinfo = xdomain.append_child("Information");
    xinfo.append_attribute("Name") = "Materials";
    String materials;
    for (Size i = 0; i < material_names.size(); ++i) {
      materials += material_names[i];
      if (i + 1 < material_names.size()) {
        materials += ", ";
      }
    }
    xinfo.append_child(pugi::node_pcdata).set_value(materials.c_str());
  }

  // Add a uniform grid
  String const h5path;
  writeXDMFUniformGrid(xdomain, h5file, h5filename, h5path, soup, name, material_names);

  // Write the XML file
  xdoc.save_file(filepath.c_str(), "  ");

  // Close the HDF5 file
  h5file.close();

  // Clang flags many string operations as potential memory leaks due to the conditional
  // free of the pointer in the long string representation. Valgrind does not report any
  // memory leaks.
  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) justified above
} // writeXDMFfile

////==============================================================================
//// addNodesToMesh
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I, std::floating_point V>
//static void
//addNodesToMesh(PolytopeSoup<T, I> & mesh, size_t const num_verts, size_t const num_dimensions,
//               H5::DataSet const & dataset, H5::FloatType const & datatype,
//               bool const xyz)
//{
//  V * data = new V[num_verts * num_dimensions];
//  dataset.read(data, datatype);
//  size_t const num_verts_old = mesh.vertices.size();
//  mesh.vertices.reserve(num_verts_old + num_verts);
//  // Add the nodes to the mesh
//  if (xyz) {
//    for (size_t i = 0; i < num_verts; ++i) {
//      T const x = static_cast<T>(data[i * 3]);
//      T const y = static_cast<T>(data[i * 3 + 1]);
//      T const z = static_cast<T>(data[i * 3 + 2]);
//      mesh.vertices.emplace_back(x, y, z);
//    }
//  } else { // XY
//    for (size_t i = 0; i < num_verts; ++i) {
//      T const x = static_cast<T>(data[i * 2]);
//      T const y = static_cast<T>(data[i * 2 + 1]);
//      T const z = 0;
//      mesh.vertices.emplace_back(x, y, z);
//    }
//  }
//  delete[] data;
//} // addNodesToMesh
//
////==============================================================================
//// readXDMFGeometry
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
//static void readXDMFGeometry(pugi::xml_node const & xgrid, H5::H5File const & h5file,
//                             std::string const & h5filename, PolytopeSoup<T, I> & mesh)
//{
//  LOG_DEBUG("Reading XDMF geometry");
//  pugi::xml_node const xgeometry = xgrid.child("Geometry");
//  if (strcmp(xgeometry.name(), "Geometry") != 0) {
//    Log::error("XDMF geometry node not found");
//    return;
//  }
//  // Get the geometry type
//  std::string const geometry_type = xgeometry.attribute("GeometryType").value();
//  if (geometry_type != "XYZ" && geometry_type != "XY") {
//    Log::error("XDMF geometry type not supported: " + String(geometry_type.c_str()));
//    return;
//  }
//  // Get the DataItem node
//  pugi::xml_node const xdataitem = xgeometry.child("DataItem");
//  if (strcmp(xdataitem.name(), "DataItem") != 0) {
//    Log::error("XDMF geometry DataItem node not found");
//    return;
//  }
//  // Get the data type
//  std::string const data_type = xdataitem.attribute("DataType").value();
//  if (data_type != "Float") {
//    Log::error("XDMF geometry data type not supported: " + String(data_type.c_str()));
//    return;
//  }
//  // Get the precision
//  std::string const precision = xdataitem.attribute("Precision").value();
//  if (precision != "4" && precision != "8") {
//    Log::error("XDMF geometry precision not supported: " + String(precision.c_str()));
//    return;
//  }
//  // Get the dimensions
//  std::string const dimensions = xdataitem.attribute("Dimensions").value();
//  size_t const split = dimensions.find_last_of(' ');
//  size_t const num_verts = std::stoul(dimensions.substr(0, split));
//  size_t const num_dimensions = std::stoul(dimensions.substr(split + 1));
//  if (geometry_type == "XYZ" && num_dimensions != 3) {
//    Log::error("XDMF geometry dimensions not supported: " + String(dimensions.c_str()));
//    return;
//  }
//  if (geometry_type == "XY" && num_dimensions != 2) {
//    Log::error("XDMF geometry dimensions not supported: " + String(dimensions.c_str()));
//    return;
//  }
//  // Get the format
//  std::string const format = xdataitem.attribute("Format").value();
//  if (format != "HDF") {
//    Log::error("XDMF geometry format not supported: " + String(format.c_str()));
//    return;
//  }
//  // Get the h5 dataset path
//  std::string const h5dataset = xdataitem.child_value();
//  // Read the data
//  H5::DataSet const dataset = h5file.openDataSet(h5dataset.substr(h5filename.size() + 1));
//#ifndef NDEBUG
//  H5T_class_t const type_class = dataset.getTypeClass();
//  assert(type_class == H5T_FLOAT);
//#endif
//  H5::FloatType const datatype = dataset.getFloatType();
//  size_t const datatype_size = datatype.getSize();
//#ifndef NDEBUG
//  assert(datatype_size == std::stoul(precision));
//  H5::DataSpace const dataspace = dataset.getSpace();
//  int const rank = dataspace.getSimpleExtentNdims();
//  assert(rank == 2);
//  hsize_t dims[2];
//  int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
//  assert(ndims == 2);
//  assert(dims[0] == num_verts);
//  assert(dims[1] == num_dimensions);
//#endif
//  if (datatype_size == 4) {
//    addNodesToMesh<T, I, float>(mesh, num_verts, num_dimensions, dataset, datatype,
//                                geometry_type == "XYZ");
//  } else if (datatype_size == 8) {
//    addNodesToMesh<T, I, double>(mesh, num_verts, num_dimensions, dataset, datatype,
//                                 geometry_type == "XYZ");
//  }
//}
//
////==============================================================================
//// addElementsToMesh
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I, std::signed_integral V>
//static void
//addElementsToMesh(size_t const num_elements, std::string const & topology_type,
//                  std::string const & dimensions, PolytopeSoup<T, I> & mesh,
//                  H5::DataSet const & dataset, H5::IntType const & datatype)
//{
//  size_t const prev_num_elements = mesh.element_types.size();
//  if (prev_num_elements == 0) {
//    mesh.element_offsets.push_back(0);
//  }
//  I const prev_offset = mesh.element_offsets.back();
//  mesh.element_offsets.insert(mesh.element_offsets.end(), num_elements, -1);
//  if (topology_type == "Mixed") {
//    mesh.element_types.insert(mesh.element_types.end(), num_elements, MeshType::None);
//    // Expect dims to be one number
//    auto const conn_length = sto<size_t>(dimensions);
//    V * data = new V[conn_length];
//    dataset.read(data, datatype);
//    // Add the elements to the mesh
//    size_t const prev_conn_size = mesh.element_conn.size();
//    size_t const num_conn_added = conn_length - num_elements;
//    mesh.element_conn.insert(mesh.element_conn.end(), num_conn_added, -1);
//    size_t offset = 0;
//    size_t position = 0;
//    for (size_t i = 0; i < num_elements; ++i) {
//      auto const element_type = static_cast<int8_t>(data[position]);
//      MeshType const mesh_type = xdmfElemTypeToMeshType(element_type);
//      mesh.element_types[prev_num_elements + i] = mesh_type;
//      auto const npoints = static_cast<size_t>(verticesPerCell(mesh_type));
//      for (size_t j = 0; j < npoints; ++j) {
//        mesh.element_conn[prev_conn_size + offset + j] =
//            static_cast<I>(static_cast<unsigned int>(data[position + j + 1]));
//      }
//      offset += npoints;
//      position += npoints + 1;
//      mesh.element_offsets[1 + prev_num_elements + i] =
//          prev_offset + static_cast<I>(offset);
//    }
//    delete[] data;
//  } else {
//    size_t const split = dimensions.find_last_of(' ');
//    auto const ncells = sto<size_t>(dimensions.substr(0, split));
//    auto const nverts = sto<size_t>(dimensions.substr(split + 1));
//    if (ncells != num_elements) {
//      Log::error("Mismatch in number of elements");
//      return;
//    }
//    V * data = new V[ncells * nverts];
//    dataset.read(data, datatype);
//    // Add the elements to the mesh
//    size_t const prev_conn_size = mesh.element_conn.size();
//    mesh.element_conn.reserve(prev_conn_size + ncells * nverts);
//    for (size_t i = 0; i < ncells; ++i) {
//      mesh.element_offsets[1 + prev_num_elements + i] =
//          static_cast<I>((i + 1U) * nverts) + prev_offset;
//      for (size_t j = 0; j < nverts; ++j) {
//        mesh.element_conn.emplace_back(static_cast<I>(data[i * nverts + j]));
//      }
//    }
//    delete[] data;
//    MeshType mesh_type = MeshType::None;
//    if (topology_type == "Triangle") {
//      mesh_type = MeshType::Tri;
//    } else if (topology_type == "Quadrilateral") {
//      mesh_type = MeshType::Quad;
//    } else if (topology_type == "Triangle_6") {
//      mesh_type = MeshType::QuadraticTri;
//    } else if (topology_type == "Quadrilateral_8") {
//      mesh_type = MeshType::QuadraticQuad;
//    } else {
//      Log::error("Unsupported element type");
//    }
//    mesh.element_types.insert(mesh.element_types.end(), ncells, mesh_type);
//  }
//}
//
////==============================================================================
//// readXDMFTopology
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
//static void readXDMFTopology(pugi::xml_node const & xgrid, H5::H5File const & h5file,
//                             std::string const & h5filename, PolytopeSoup<T, I> & mesh)
//{
//  LOG_DEBUG("Reading XDMF topology");
//  pugi::xml_node const xtopology = xgrid.child("Topology");
//  if (strcmp(xtopology.name(), "Topology") != 0) {
//    Log::error("XDMF topology node not found");
//    return;
//  }
//  // Get the topology type
//  std::string const topology_type = xtopology.attribute("TopologyType").value();
//  // Get the number of elements
//  size_t const num_elements = std::stoul(xtopology.attribute("NumberOfElements").value());
//  // Get the DataItem node
//  pugi::xml_node const xdataitem = xtopology.child("DataItem");
//  if (strcmp(xdataitem.name(), "DataItem") != 0) {
//    Log::error("XDMF topology DataItem node not found");
//    return;
//  }
//  // Get the data type
//  std::string const data_type = xdataitem.attribute("DataType").value();
//  if (data_type != "Int") {
//    Log::error("XDMF topology data type not supported: " + String(data_type.c_str()));
//    return;
//  }
//  // Get the precision
//  std::string const precision = xdataitem.attribute("Precision").value();
//  if (precision != "1" && precision != "2" && precision != "4" && precision != "8") {
//    Log::error("XDMF topology precision not supported: " + String(precision.c_str()));
//    return;
//  }
//  // Get the format
//  std::string const format = xdataitem.attribute("Format").value();
//  if (format != "HDF") {
//    Log::error("XDMF geometry format not supported: " + String(format.c_str()));
//    return;
//  }
//  // Get the h5 dataset path
//  std::string const h5dataset = xdataitem.child_value();
//  // Read the data
//  H5::DataSet const dataset = h5file.openDataSet(h5dataset.substr(h5filename.size() + 1));
//#ifndef NDEBUG
//  H5T_class_t const type_class = dataset.getTypeClass();
//  assert(type_class == H5T_INTEGER);
//#endif
//  H5::IntType const datatype = dataset.getIntType();
//  size_t const datatype_size = datatype.getSize();
//#ifndef NDEBUG
//  assert(datatype_size == std::stoul(precision));
//  H5::DataSpace const dataspace = dataset.getSpace();
//  int const rank = dataspace.getSimpleExtentNdims();
//  if (topology_type == "Mixed") {
//    assert(rank == 1);
//    hsize_t dims[1];
//    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
//    assert(ndims == 1);
//  } else {
//    assert(rank == 2);
//    hsize_t dims[2];
//    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
//    assert(ndims == 2);
//  }
//#endif
//  // Get the dimensions
//  std::string const dimensions = xdataitem.attribute("Dimensions").value();
//  if (datatype_size == 1) {
//    addElementsToMesh<T, I, int8_t>(num_elements, topology_type, dimensions, mesh,
//                                    dataset, datatype);
//  } else if (datatype_size == 2) {
//    addElementsToMesh<T, I, int16_t>(num_elements, topology_type, dimensions, mesh,
//                                     dataset, datatype);
//  } else if (datatype_size == 4) {
//    addElementsToMesh<T, I, int32_t>(num_elements, topology_type, dimensions, mesh,
//                                     dataset, datatype);
//  } else if (datatype_size == 8) {
//    addElementsToMesh<T, I, int64_t>(num_elements, topology_type, dimensions, mesh,
//                                     dataset, datatype);
//  } else {
//    Log::error("Unsupported data type size");
//  }
//}
//
////==============================================================================
//// addElsetToMesh
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I, std::signed_integral V>
//static void
//addElsetToMesh(PolytopeSoup<T, I> & mesh, size_t const num_elements,
//               H5::DataSet const & dataset, H5::IntType const & datatype)
//{
//  V * data = new V[num_elements];
//  dataset.read(data, datatype);
//  I const last_offset = mesh.elset_offsets.back();
//  mesh.elset_offsets.push_back(last_offset + static_cast<I>(num_elements));
//  mesh.elset_ids.reserve(mesh.elset_ids.size() + num_elements);
//  for (size_t i = 0; i < num_elements; ++i) {
//    mesh.elset_ids.emplace_back(static_cast<I>(data[i]));
//  }
//  delete[] data;
//}
//
////==============================================================================
//// readXDMFElsets
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
//static void readXDMFElsets(pugi::xml_node const & xgrid, H5::H5File const & h5file,
//                           std::string const & h5filename, PolytopeSoup<T, I> & mesh)
//{
//  LOG_DEBUG("Reading XDMF elsets");
//  // Loop over all nodes to find the elsets
//  for (pugi::xml_node xelset = xgrid.first_child(); xelset;
//       xelset = xelset.next_sibling()) {
//    if (strcmp(xelset.name(), "Set") != 0) {
//      continue;
//    }
//    // Get the SetType
//    std::string const set_type = xelset.attribute("SetType").value();
//    if (set_type != "Cell") {
//      Log::error("XDMF elset only supports SetType=Cell");
//      return;
//    }
//    // Get the name
//    std::string const name = xelset.attribute("Name").value();
//    if (name.empty()) {
//      Log::error("XDMF elset name not found");
//      return;
//    }
//    // Get the DataItem node
//    pugi::xml_node const xdataitem = xelset.child("DataItem");
//    if (strcmp(xdataitem.name(), "DataItem") != 0) {
//      Log::error("XDMF elset DataItem node not found");
//      return;
//    }
//    // Get the data type
//    std::string const data_type = xdataitem.attribute("DataType").value();
//    if (data_type != "Int") {
//      Log::error("XDMF elset data type not supported: " + String(data_type.c_str()));
//      return;
//    }
//    // Get the precision
//    std::string const precision = xdataitem.attribute("Precision").value();
//    if (precision != "1" && precision != "2" && precision != "4" && precision != "8") {
//      Log::error("XDMF elset precision not supported: " + String(precision.c_str()));
//      return;
//    }
//    // Get the format
//    std::string const format = xdataitem.attribute("Format").value();
//    if (format != "HDF") {
//      Log::error("XDMF elset format not supported: " + String(format.c_str()));
//      return;
//    }
//    // Get the h5 dataset path
//    std::string const h5dataset = xdataitem.child_value();
//    // Read the data
//    H5::DataSet const dataset =
//        h5file.openDataSet(h5dataset.substr(h5filename.size() + 1));
//#ifndef NDEBUG
//    H5T_class_t const type_class = dataset.getTypeClass();
//    assert(type_class == H5T_INTEGER);
//#endif
//    H5::IntType const datatype = dataset.getIntType();
//    size_t const datatype_size = datatype.getSize();
//    assert(datatype_size == std::stoul(precision));
//    H5::DataSpace const dataspace = dataset.getSpace();
//#ifndef NDEBUG
//    int const rank = dataspace.getSimpleExtentNdims();
//    assert(rank == 1);
//#endif
//    hsize_t dims[1];
//#ifndef NDEBUG
//    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
//    assert(ndims == 1);
//#else
//    dataspace.getSimpleExtentDims(dims, nullptr);
//#endif
//    // Get the dimensions
//    std::string const dimensions = xdataitem.attribute("Dimensions").value();
//    size_t const num_elements = dims[0];
//    assert(num_elements == std::stoul(dimensions));
//    mesh.elset_names.push_back(name);
//    if (mesh.elset_offsets.empty()) {
//      mesh.elset_offsets.push_back(0);
//    }
//    if (datatype_size == 1) {
//      addElsetToMesh<T, I, int8_t>(mesh, num_elements, dataset, datatype);
//    } else if (datatype_size == 2) {
//      addElsetToMesh<T, I, int16_t>(mesh, num_elements, dataset, datatype);
//    } else if (datatype_size == 4) {
//      addElsetToMesh<T, I, int32_t>(mesh, num_elements, dataset, datatype);
//    } else if (datatype_size == 8) {
//      addElsetToMesh<T, I, int64_t>(mesh, num_elements, dataset, datatype);
//    }
//  }
//}
//
////==============================================================================
//// readXDMFUniformGrid
////==============================================================================
//
//template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
//void readXDMFUniformGrid(pugi::xml_node const & xgrid, H5::H5File const & h5file,
//                         std::string const & h5filename,
//                         //    std::vector<std::string> const & material_names,
//                         PolytopeSoup<T, I> & mesh)
//{
//  readXDMFGeometry(xgrid, h5file, h5filename, mesh);
//  readXDMFTopology(xgrid, h5file, h5filename, mesh);
//  readXDMFElsets(xgrid, h5file, h5filename, mesh);
//}
//
//==============================================================================
// readXDMFFile
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
readXDMFFile(String const & filename, PolytopeSoup<T, I> & /*soup*/)
{
  Log::info("Reading XDMF mesh file: " + filename);
//
//  // Open HDF5 file
//  size_t const h5filepath_end = filename.find_last_of('/') + 1;
//  std::string const h5filename =
//      filename.substr(h5filepath_end, filename.size() - 4 - h5filepath_end) + "h5";
//  std::string const h5filepath = filename.substr(0, h5filepath_end);
//  LOG_DEBUG("H5 filename: " + String(h5filename.c_str()));
//  H5::H5File const h5file(h5filepath + h5filename, H5F_ACC_RDONLY);
//
//  // Setup XML file
//  pugi::xml_document xdoc;
//  pugi::xml_parse_result const result = xdoc.load_file(filename.c_str());
//  if (!result) {
//    Log::error("XDMF XML parse error: " + String(result.description()) +
//               ", character pos= " + toString(result.offset));
//  }
//  pugi::xml_node const xroot = xdoc.child("Xdmf");
//  if (strcmp("Xdmf", xroot.name()) != 0) {
//    Log::error("XDMF XML root node is not Xdmf");
//    return;
//  }
//  pugi::xml_node const xdomain = xroot.child("Domain");
//  if (strcmp("Domain", xdomain.name()) != 0) {
//    Log::error("XDMF XML domain node is not Domain");
//    return;
//  }
//
//  // pugi::xml_node const xinfo = xdomain.child("Information");
//  // std::vector<std::string> material_names;
//  // if (strcmp("Information", xinfo.name()) == 0) {
//  //   // Get the "Name" attribute
//  //   pugi::xml_attribute const xname = xinfo.attribute("Name");
//  //   if (strcmp("Materials", xname.value()) == 0) {
//  //     // Get the material names
//  //     std::string const materials = xinfo.child_value();
//  //     std::stringstream ss(materials);
//  //     std::string material;
//  //     while (std::getline(ss, material, ',')) {
//  //       if (material[0] == ' ') {
//  //         material_names.push_back(material.substr(1));
//  //       } else {
//  //         material_names.push_back(material);
//  //       }
//  //     }
//  //   }
//  // }
//
//  pugi::xml_node const xgrid = xdomain.child("Grid");
//  if (strcmp("Grid", xgrid.name()) != 0) {
//    Log::error("XDMF XML grid node is not Grid");
//    return;
//  }
//  mesh.name = xgrid.attribute("Name").value();
//  if (strcmp("Uniform", xgrid.attribute("GridType").value()) == 0) {
//    readXDMFUniformGrid(xgrid, h5file, h5filename, mesh);
//  } else if (strcmp("Tree", xgrid.attribute("GridType").value()) == 0) {
//    Log::error("XDMF XML Tree is not supported");
//  } else {
//    Log::error("XDMF XML grid type is not Uniform or Tree");
//  }
}

} // namespace um2
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
