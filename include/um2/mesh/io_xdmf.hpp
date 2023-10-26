#pragma once

#include <um2/mesh/PolytopeSoup.hpp>

#include <um2/stdlib/sto.hpp>

// #include <cstring>   // strcmp
// #include <sstream>   // std::stringstream

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

//==============================================================================
// writeXDMFGeometry
//==============================================================================

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void writeXDMFGeometry(pugi::xml_node & xgrid, H5::Group & h5group,
                              String const & h5filename, String const & h5path,
                              PolytopeSoup<T, I> const & soup)
{
  LOG_DEBUG("Writing XDMF geometry");
  Size const num_verts = soup.vertices.size();
  bool const is_2d =
      std::count_if(soup.vertices.cbegin(), soup.vertices.cend(), [](auto const & v) {
        return um2::abs(v[2]) < eps_distance<T>;
      }) == static_cast<int64_t>(num_verts);
  Size const dim = is_2d ? 2 : 3;
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
      (toString(num_verts) + " " + toString(dim)).c_str();
  xdata.append_attribute("Precision") = sizeof(T);
  xdata.append_attribute("Format") = "HDF";
  String const h5geompath = h5filename + ":" + h5path + "/Geometry";
  xdata.append_child(pugi::node_pcdata).set_value(h5geompath.c_str());

  // Create HDF5 data space
  hsize_t dims[2] = {static_cast<hsize_t>(num_verts), static_cast<hsize_t>(dim)};
  H5::DataSpace const h5space(2, dims);
  // Create HDF5 data type
  H5::DataType const h5type = getH5DataType<T>();
  // Create HDF5 data set
  H5::DataSet const h5dataset = h5group.createDataSet("Geometry", h5type, h5space);
  // Create an xy or xyz array
  Vector<T> xyz(num_verts * dim);
  if (dim == 2) {
    for (Size i = 0; i < num_verts; ++i) {
      xyz[2 * i] = soup.vertices[i][0];
      xyz[2 * i + 1] = soup.vertices[i][1];
    }
  } else { // dim == 3
    for (Size i = 0; i < num_verts; ++i) {
      xyz[3 * i] = soup.vertices[i][0];
      xyz[3 * i + 1] = soup.vertices[i][1];
      xyz[3 * i + 2] = soup.vertices[i][2];
    }
  }
  // Write HDF5 data set
  h5dataset.write(xyz.data(), h5type, h5space);
} // writeXDMFgeometry

//==============================================================================
// writeXDMFTopology
//==============================================================================

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void writeXDMFTopology(pugi::xml_node & xgrid, H5::Group & h5group,
                              String const & h5filename, String const & h5path,
                              PolytopeSoup<T, I> const & soup)
{
  LOG_DEBUG("Writing XDMF topology");
  // Create XDMF Topology node
  auto xtopo = xgrid.append_child("Topology");
  Size const nelems = soup.numElems();

  Vector<I> topology;
  String topology_type;
  String dimensions;
  Size nverts = 0;
  auto const elem_type = soup.getElemTypes();
  VTKElemType const type0 = elem_type[0];
  VTKElemType const type1 = elem_type[1];
  bool ishomogeneous = true;
  if (type1 == VTKElemType::None) {
    switch (type0) {
    case VTKElemType::Triangle:
      topology_type = "Triangle";
      nverts = 3;
      break;
    case VTKElemType::Quad:
      topology_type = "Quadrilateral";
      nverts = 4;
      break;
    case VTKElemType::QuadraticEdge:
      topology_type = "Edge_3";
      nverts = 3;
      break;
    case VTKElemType::QuadraticTriangle:
      topology_type = "Triangle_6";
      nverts = 6;
      break;
    case VTKElemType::QuadraticQuad:
      topology_type = "Quadrilateral_8";
      nverts = 8;
      break;
    default:
      Log::error("Unsupported polytope type");
    }
    dimensions = toString(nelems) + " " + toString(nverts);
  } else {
    topology_type = "Mixed";
    ishomogeneous = false;
    dimensions = toString(nelems + soup.element_conn.size());
    topology.resize(nelems + soup.element_conn.size());
    // Create the topology array (type id + node ids)
    Size topo_ctr = 0;
    for (Size i = 0; i < nelems; ++i) {
      int8_t const topo_type = vtkToXDMFElemType(soup.element_types[i]);
      if (topo_type == -1) {
        Log::error("Unsupported polytope type");
      }
      topology[topo_ctr] = static_cast<I>(static_cast<unsigned int>(topo_type));
      auto const offset = static_cast<Size>(soup.element_offsets[i]);
      auto const npts =
          static_cast<Size>(soup.element_offsets[i + 1] - soup.element_offsets[i]);
      for (Size j = 0; j < npts; ++j) {
        topology[topo_ctr + j + 1] = soup.element_conn[offset + j];
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
  xdata.append_attribute("Precision") = sizeof(I);
  xdata.append_attribute("Format") = "HDF";
  String const h5topopath = h5filename + ":" + h5path + "/Topology";
  xdata.append_child(pugi::node_pcdata).set_value(h5topopath.c_str());

  // Create HDF5 data type
  H5::DataType const h5type = getH5DataType<I>();
  if (ishomogeneous) {
    // Create HDF5 data space
    hsize_t dims[2] = {static_cast<hsize_t>(nelems), static_cast<hsize_t>(nverts)};
    H5::DataSpace const h5space(2, dims);
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
    // Write HDF5 data set
    h5dataset.write(soup.element_conn.data(), h5type, h5space);
  } else {
    // Create HDF5 data space
    auto const dims = static_cast<hsize_t>(topology.size());
    H5::DataSpace const h5space(1, &dims);
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet("Topology", h5type, h5space);
    // Write HDF5 data set
    h5dataset.write(topology.data(), h5type, h5space);
  }
} // writeXDMFTopology

////==============================================================================
//// writeXDMFMaterials
////==============================================================================
//
// template <std::floating_point T, std::signed_integral I>
//  requires(sizeof(T) == 4 || sizeof(T) == 8)
// static void writeXDMFMaterials(pugi::xml_node & xgrid, H5::Group & h5group,
//                               std::string const & h5filename, std::string const &
//                               h5path, PolytopeSoup<T, I> const & mesh,
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

//==============================================================================
// writeXDMFElsets
//==============================================================================

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void writeXDMFElsets(pugi::xml_node & xgrid, H5::Group & h5group,
                            String const & h5filename, String const & h5path,
                            PolytopeSoup<T, I> const & soup,
                            Vector<String> const & material_names)
{
  LOG_DEBUG("Writing XDMF elsets");
  for (Size i = 0; i < soup.elset_names.size(); ++i) {
    String const name = soup.elset_names[i];
    auto const start = static_cast<Size>(soup.elset_offsets[i]);
    auto const end = static_cast<Size>(soup.elset_offsets[i + 1]);
    // Create HDF5 data space
    auto dims = static_cast<hsize_t>(end - start);
    H5::DataSpace const h5space(1, &dims);
    // Create HDF5 data type
    H5::DataType const h5type = getH5DataType<I>();
    // Create HDF5 data set
    H5::DataSet const h5dataset = h5group.createDataSet(name.c_str(), h5type, h5space);
    // Write HDF5 data set.
    h5dataset.write(&soup.elset_ids[start], h5type, h5space);

    // Create XDMF Elset node
    auto xelset = xgrid.append_child("Set");
    xelset.append_attribute("Name") = name.c_str();
    xelset.append_attribute("SetType") = "Cell";
    // Create XDMF DataItem node
    auto xdata = xelset.append_child("DataItem");
    xdata.append_attribute("DataType") = "Int";
    xdata.append_attribute("Dimensions") = end - start;
    xdata.append_attribute("Precision") = sizeof(I);
    xdata.append_attribute("Format") = "HDF";
    String h5elsetpath = h5filename;
    h5elsetpath += ':';
    h5elsetpath += h5path;
    h5elsetpath += '/';
    h5elsetpath += name;
    xdata.append_child(pugi::node_pcdata).set_value(h5elsetpath.c_str());

    if (!soup.elset_data[i].empty()) {
      if (soup.elset_names[i].starts_with("Material_")) {
        Log::error("Material elsets should not have data");
      }
      // Create HDF5 data space
      auto const dims_data = static_cast<hsize_t>(soup.elset_data[i].size());
      H5::DataSpace const h5space_data(1, &dims_data);
      // Create HDF5 data type
      H5::DataType const h5type_data = getH5DataType<T>();
      // Create HDF5 data set
      H5::DataSet const h5dataset_data =
          h5group.createDataSet((name + "_data").c_str(), h5type_data, h5space_data);
      // Write HDF5 data set
      h5dataset_data.write(soup.elset_data[i].data(), h5type_data, h5space_data);

      // Create XDMF data node
      auto xatt = xelset.append_child("Attribute");
      xatt.append_attribute("Name") = (name + "_data").c_str();
      xatt.append_attribute("Center") = "Cell";
      // Create XDMF DataItem node
      auto xdata2 = xatt.append_child("DataItem");
      xdata2.append_attribute("DataType") = "Float";
      xdata2.append_attribute("Dimensions") = soup.elset_data[i].size();
      xdata2.append_attribute("Precision") = sizeof(T);
      xdata2.append_attribute("Format") = "HDF";

      String const h5elsetdatapath = h5elsetpath + "_data";
      xdata2.append_child(pugi::node_pcdata).set_value(h5elsetdatapath.c_str());
    }

    if (name.starts_with("Material_")) {
      // Get the index of the material in the material_names vector
      Size index = -1;
      for (Size j = 0; j < material_names.size(); ++j) {
        if (name == material_names[j]) {
          index = j;
          break;
        }
      }
      if (index == -1) {
        Log::error("Could not find material name in material_names vector");
      }

      // Create HDF5 data space
      auto const dims_data = static_cast<hsize_t>(end - start);
      H5::DataSpace const h5space_data(1, &dims_data);
      // Create HDF5 data type
      H5::DataType const h5type_data = getH5DataType<int>();
      // Create HDF5 data set
      H5::DataSet const h5dataset_data =
          h5group.createDataSet((name + "_data").c_str(), h5type_data, h5space_data);
      Vector<int> material_ids(end - start, index);
      // Write HDF5 data set
      h5dataset_data.write(material_ids.data(), h5type_data, h5space_data);

      // Create XDMF data node
      auto xatt = xelset.append_child("Attribute");
      xatt.append_attribute("Name") = "Material";
      xatt.append_attribute("Center") = "Cell";
      // Create XDMF DataItem node
      auto xdata2 = xatt.append_child("DataItem");
      xdata2.append_attribute("DataType") = "Int";
      xdata2.append_attribute("Dimensions") = material_ids.size();
      xdata2.append_attribute("Precision") = sizeof(int);
      xdata2.append_attribute("Format") = "HDF";

      String const h5elsetdatapath = h5elsetpath + "_data";
      xdata2.append_child(pugi::node_pcdata).set_value(h5elsetdatapath.c_str());
    }
  }
} // writeXDMFelsets

//==============================================================================
// writeXDMFUniformGrid
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
writeXDMFUniformGrid(pugi::xml_node & xdomain, H5::H5File & h5file,
                     String const & h5filename, String const & h5path,
                     PolytopeSoup<T, I> const & soup, String const & name,
                     Vector<String> const & material_names)
{
  LOG_DEBUG("Writing XDMF uniform grid");

  // Grid
  pugi::xml_node xgrid = xdomain.append_child("Grid");
  xgrid.append_attribute("Name") = name.c_str();
  xgrid.append_attribute("GridType") = "Uniform";

  // h5
  String const h5grouppath = h5path + "/" + name;
  H5::Group h5group = h5file.createGroup(h5grouppath.c_str());

  writeXDMFGeometry(xgrid, h5group, h5filename, h5grouppath, soup);
  writeXDMFTopology(xgrid, h5group, h5filename, h5grouppath, soup);
  writeXDMFElsets(xgrid, h5group, h5filename, h5grouppath, soup, material_names);
} // writeXDMFUniformGrid

//==============================================================================
// writeXDMFFile
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
writeXDMFFile(String const & filepath, PolytopeSoup<T, I> & soup,
              String const & name = "UM2")
{

  Log::info("Writing XDMF file: " + filepath);

  // Setup HDF5 file
  // Get the h5 file name
  Size const h5filepath_end = filepath.find_last_of('/') + 1;
  String const h5filename =
      filepath.substr(h5filepath_end, filepath.size() - 5 - h5filepath_end) + ".h5";
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

  // If there are any materials, add an information node listing them
  if (!material_names.empty()) {
    pugi::xml_node xinfo = xdomain.append_child("Information");
    xinfo.append_attribute("Name") = "Materials";
    String materials;
    for (Size i = 0; i < material_names.size(); ++i) {
      auto const & mat_name = material_names[i];
      String const short_name = mat_name.substr(9, mat_name.size() - 9);
      materials += short_name;
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
} // writeXDMFfile

//==============================================================================
// addNodesToMesh
//==============================================================================

template <std::floating_point T, std::signed_integral I, std::floating_point V>
static void
addNodesToMesh(PolytopeSoup<T, I> & mesh, Size const num_verts, Size const num_dimensions,
               H5::DataSet const & dataset, H5::FloatType const & datatype,
               bool const xyz)
{
  Vector<V> data_vec(num_verts * num_dimensions);
  dataset.read(data_vec.data(), datatype);
  Size const num_verts_old = mesh.vertices.size();
  mesh.vertices.resize(num_verts_old + num_verts);
  // Add the nodes to the mesh
  if (xyz) {
    for (Size i = 0; i < num_verts; ++i) {
      T const x = static_cast<T>(data_vec[i * 3]);
      T const y = static_cast<T>(data_vec[i * 3 + 1]);
      T const z = static_cast<T>(data_vec[i * 3 + 2]);
      mesh.vertices[num_verts_old + i] = Point3<T>(x, y, z);
    }
  } else { // XY
    for (Size i = 0; i < num_verts; ++i) {
      T const x = static_cast<T>(data_vec[i * 2]);
      T const y = static_cast<T>(data_vec[i * 2 + 1]);
      T const z = 0;
      mesh.vertices[num_verts_old + i] = Point3<T>(x, y, z);
    }
  }
} // addNodesToMesh

//==============================================================================
// readXDMFGeometry
//==============================================================================

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
static void readXDMFGeometry(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                             String const & h5filename, PolytopeSoup<T, I> & soup)
{
  LOG_DEBUG("Reading XDMF geometry");
  pugi::xml_node const xgeometry = xgrid.child("Geometry");
  if (strcmp(xgeometry.name(), "Geometry") != 0) {
    Log::error("XDMF geometry node not found");
    return;
  }
  // Get the geometry type
  String const geometry_type(xgeometry.attribute("GeometryType").value());
  if (geometry_type != "XYZ" && geometry_type != "XY") {
    Log::error("XDMF geometry type not supported: " + geometry_type);
    return;
  }
  // Get the DataItem node
  pugi::xml_node const xdataitem = xgeometry.child("DataItem");
  if (strcmp(xdataitem.name(), "DataItem") != 0) {
    Log::error("XDMF geometry DataItem node not found");
    return;
  }
  // Get the data type
  String const data_type(xdataitem.attribute("DataType").value());
  if (data_type != "Float") {
    Log::error("XDMF geometry data type not supported: " + data_type);
    return;
  }
  // Get the precision
  std::string const precision(xdataitem.attribute("Precision").value());
  if (precision != "4" && precision != "8") {
    Log::error("XDMF geometry precision not supported: " + String(precision.c_str()));
    return;
  }
  // Get the dimensions
  std::string const dimensions(xdataitem.attribute("Dimensions").value());
  size_t const split = dimensions.find_last_of(' ');
  Size const num_verts = sto<Size>(dimensions.substr(0, split));
  Size const num_dimensions = sto<Size>(dimensions.substr(split + 1));
  if (geometry_type == "XYZ" && num_dimensions != 3) {
    Log::error("XDMF geometry dimensions not supported: " + String(dimensions.c_str()));
    return;
  }
  if (geometry_type == "XY" && num_dimensions != 2) {
    Log::error("XDMF geometry dimensions not supported: " + String(dimensions.c_str()));
    return;
  }
  // Get the format
  String const format(xdataitem.attribute("Format").value());
  if (format != "HDF") {
    Log::error("XDMF geometry format not supported: " + format);
    return;
  }


  // Get the h5 dataset path
  String const h5dataset(xdataitem.child_value());
  // Read the data
  H5::DataSet const dataset = h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).c_str());
#ifndef NDEBUG
  H5T_class_t const type_class = dataset.getTypeClass();
  assert(type_class == H5T_FLOAT);
#endif
  H5::FloatType const datatype = dataset.getFloatType();
  size_t const datatype_size = datatype.getSize();
#ifndef NDEBUG
  assert(datatype_size == std::stoul(precision));
  H5::DataSpace const dataspace = dataset.getSpace();
  int const rank = dataspace.getSimpleExtentNdims();
  assert(rank == 2);
  hsize_t dims[2];
  int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
  assert(ndims == 2);
  assert(dims[0] == static_cast<hsize_t>(num_verts));
  assert(dims[1] == static_cast<hsize_t>(num_dimensions));
#endif
  if (datatype_size == 4) {
    addNodesToMesh<T, I, float>(soup, num_verts, num_dimensions, dataset, datatype,
                                geometry_type == "XYZ");
  } else if (datatype_size == 8) {
    addNodesToMesh<T, I, double>(soup, num_verts, num_dimensions, dataset, datatype,
                                 geometry_type == "XYZ");
  }
}

//==============================================================================
// addElementsToMesh
//==============================================================================

 template <std::floating_point T, std::signed_integral I, std::signed_integral V>
 static void
 addElementsToMesh(Size const num_elements, String const & topology_type,
                  std::string const & dimensions, PolytopeSoup<T, I> & soup,
                  H5::DataSet const & dataset, H5::IntType const & datatype)
{
  Size const prev_num_elements = soup.element_types.size();
  if (prev_num_elements == 0) {
    soup.element_offsets.push_back(0);
  }
  I const prev_offset = soup.element_offsets.back();
  soup.element_offsets.push_back(num_elements, -1);
  if (topology_type == "Mixed") {
    soup.element_types.push_back(num_elements, VTKElemType::None);
    // Expect dims to be one number
    auto const conn_length = sto<Size>(dimensions);
    Vector<V> data_vec(conn_length);
    dataset.read(data_vec.data(), datatype);
    // Add the elements to the soup
    Size const prev_conn_size = soup.element_conn.size();
    Size const num_conn_added = conn_length - num_elements;
    soup.element_conn.push_back(num_conn_added, -1);
    Size offset = 0;
    Size position = 0;
    for (Size i = 0; i < num_elements; ++i) {
      auto const element_type = static_cast<int8_t>(data_vec[position]);
      VTKElemType const elem_type = xdmfToVTKElemType(element_type);
      soup.element_types[prev_num_elements + i] = elem_type;
      auto const npoints = verticesPerElem(elem_type);
      for (Size j = 0; j < npoints; ++j) {
        soup.element_conn[prev_conn_size + offset + j] =
            static_cast<I>(static_cast<unsigned int>(data_vec[position + j + 1]));
      }
      offset += npoints;
      position += npoints + 1;
      soup.element_offsets[1 + prev_num_elements + i] =
          prev_offset + static_cast<I>(offset);
    }
  } else {
    size_t const split = dimensions.find_last_of(' ');
    auto const ncells = sto<Size>(dimensions.substr(0, split));
    auto const nverts = sto<Size>(dimensions.substr(split + 1));
    if (ncells != num_elements) {
      Log::error("Mismatch in number of elements");
      return;
    }
    Vector<V> data_vec(ncells * nverts);
    dataset.read(data_vec.data(), datatype);
    // Add the elements to the soup
    for (Size i = 0; i < ncells; ++i) {
      soup.element_offsets[1 + prev_num_elements + i] =
          static_cast<I>((i + 1) * nverts) + prev_offset;
      for (Size j = 0; j < nverts; ++j) {
        soup.element_conn.push_back(static_cast<I>(data_vec[i * nverts + j]));
      }
    }
    VTKElemType elem_type = VTKElemType::None;
    if (topology_type == "Triangle") {
      elem_type = VTKElemType::Triangle;
    } else if (topology_type == "Quadrilateral") {
      elem_type = VTKElemType::Quad;
    } else if (topology_type == "Triangle_6") {
      elem_type = VTKElemType::QuadraticTriangle;
    } else if (topology_type == "Quadrilateral_8") {
      elem_type = VTKElemType::QuadraticQuad;
    } else {
      Log::error("Unsupported element type");
    }
    soup.element_types.push_back(ncells, elem_type);
  }
}

//==============================================================================
// readXDMFTopology
//==============================================================================

 template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
 static void readXDMFTopology(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                             String const & h5filename, PolytopeSoup<T, I> & soup)
{
  LOG_DEBUG("Reading XDMF topology");
  pugi::xml_node const xtopology = xgrid.child("Topology");
  if (strcmp(xtopology.name(), "Topology") != 0) {
    Log::error("XDMF topology node not found");
    return;
  }
  // Get the topology type
  String const topology_type(xtopology.attribute("TopologyType").value());
  // Get the number of elements
  Size const num_elements = sto<Size>(xtopology.attribute("NumberOfElements").value());
  // Get the DataItem node
  pugi::xml_node const xdataitem = xtopology.child("DataItem");
  if (strcmp(xdataitem.name(), "DataItem") != 0) {
    Log::error("XDMF topology DataItem node not found");
    return;
  }
  // Get the data type
  String const data_type(xdataitem.attribute("DataType").value());
  if (data_type != "Int") {
    Log::error("XDMF topology data type not supported: " + data_type);
    return;
  }
  // Get the precision
  std::string const precision(xdataitem.attribute("Precision").value());
  if (precision != "1" && precision != "2" && precision != "4" && precision != "8") {
    Log::error("XDMF topology precision not supported: " + String(precision.c_str()));
    return;
  }
  // Get the format
  String const format(xdataitem.attribute("Format").value());
  if (format != "HDF") {
    Log::error("XDMF geometry format not supported: " + format);
    return;
  }
  // Get the h5 dataset path
  String const h5dataset(xdataitem.child_value());
  // Read the data
  H5::DataSet const dataset = h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).c_str());
#ifndef NDEBUG
  H5T_class_t const type_class = dataset.getTypeClass();
  assert(type_class == H5T_INTEGER);
#endif
  H5::IntType const datatype = dataset.getIntType();
  size_t const datatype_size = datatype.getSize();
#ifndef NDEBUG
  assert(datatype_size == std::stoul(precision));
  H5::DataSpace const dataspace = dataset.getSpace();
  int const rank = dataspace.getSimpleExtentNdims();
  if (topology_type == "Mixed") {
    assert(rank == 1);
    hsize_t dims[1];
    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
    assert(ndims == 1);
  } else {
    assert(rank == 2);
    hsize_t dims[2];
    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
    assert(ndims == 2);
  }
#endif
  // Get the dimensions
  std::string const dimensions = xdataitem.attribute("Dimensions").value();
  if (datatype_size == 1) {
    addElementsToMesh<T, I, int8_t>(num_elements, topology_type, dimensions, soup,
                                    dataset, datatype);
  } else if (datatype_size == 2) {
    addElementsToMesh<T, I, int16_t>(num_elements, topology_type, dimensions, soup,
                                     dataset, datatype);
  } else if (datatype_size == 4) {
    addElementsToMesh<T, I, int32_t>(num_elements, topology_type, dimensions, soup,
                                     dataset, datatype);
  } else if (datatype_size == 8) {
    addElementsToMesh<T, I, int64_t>(num_elements, topology_type, dimensions, soup,
                                     dataset, datatype);
  } else {
    Log::error("Unsupported data type size");
  }
}

//==============================================================================
// addElsetToMesh
//==============================================================================

 template <std::floating_point T, std::signed_integral I, std::signed_integral V>
 static void
 addElsetToMesh(PolytopeSoup<T, I> & soup, Size const num_elements,
               H5::DataSet const & dataset, H5::IntType const & datatype, String const & elset_name)
{
  Vector<V> data_vec(num_elements);
  dataset.read(data_vec.data(), datatype);
  Vector<I> elset_ids(num_elements);
  for (Size i = 0; i < num_elements; ++i) {
    elset_ids[i] = static_cast<I>(static_cast<uint32_t>(data_vec[i]));
  }
  soup.addElset(elset_name, elset_ids);
}

//==============================================================================
// readXDMFElsets
//==============================================================================

 template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
 static void readXDMFElsets(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                           String const & h5filename, PolytopeSoup<T, I> & soup)
{
  LOG_DEBUG("Reading XDMF elsets");
  // Loop over all nodes to find the elsets
  for (pugi::xml_node xelset = xgrid.first_child(); xelset; xelset = xelset.next_sibling()) {
    if (strcmp(xelset.name(), "Set") != 0) {
      continue;
    }
    // Get the SetType
    String const set_type(xelset.attribute("SetType").value());
    if (set_type != "Cell") {
      Log::error("XDMF elset only supports SetType=Cell");
      return;
    }
    // Get the name
    String const name(xelset.attribute("Name").value());
    if (name.size() == 0) {
      Log::error("XDMF elset name not found");
      return;
    }
    // Get the DataItem node
    pugi::xml_node const xdataitem = xelset.child("DataItem");
    if (strcmp(xdataitem.name(), "DataItem") != 0) {
      Log::error("XDMF elset DataItem node not found");
      return;
    }
    // Get the data type
    String const data_type(xdataitem.attribute("DataType").value());
    if (data_type != "Int") {
      Log::error("XDMF elset data type not supported: " + data_type);
      return;
    }
    // Get the precision
    std::string const precision = xdataitem.attribute("Precision").value();
    if (precision != "1" && precision != "2" && precision != "4" && precision != "8") {
      Log::error("XDMF elset precision not supported: " + String(precision.c_str()));
      return;
    }
    // Get the format
    String const format(xdataitem.attribute("Format").value());
    if (format != "HDF") {
      Log::error("XDMF elset format not supported: " + format);
      return;
    }
    // Get the h5 dataset path
    String const h5dataset(xdataitem.child_value());
    // Read the data
    H5::DataSet const dataset =
        h5file.openDataSet(h5dataset.substr(h5filename.size() + 1).c_str());
 #ifndef NDEBUG
    H5T_class_t const type_class = dataset.getTypeClass();
    assert(type_class == H5T_INTEGER);
 #endif
    H5::IntType const datatype = dataset.getIntType();
    size_t const datatype_size = datatype.getSize();
    assert(datatype_size == std::stoul(precision));
    H5::DataSpace const dataspace = dataset.getSpace();
 #ifndef NDEBUG
    int const rank = dataspace.getSimpleExtentNdims();
    assert(rank == 1);
 #endif
    hsize_t dims[1];
 #ifndef NDEBUG
    int const ndims = dataspace.getSimpleExtentDims(dims, nullptr);
    assert(ndims == 1);
 #else
    dataspace.getSimpleExtentDims(dims, nullptr);
 #endif
    // Get the dimensions
    std::string const dimensions = xdataitem.attribute("Dimensions").value();
    auto const num_elements = static_cast<Size>(dims[0]);
    assert(num_elements == sto<Size>(dimensions));
    if (datatype_size == 1) {
      addElsetToMesh<T, I, int8_t>(soup, num_elements, dataset, datatype, name);
    } else if (datatype_size == 2) {
      addElsetToMesh<T, I, int16_t>(soup, num_elements, dataset, datatype, name);
    } else if (datatype_size == 4) {
      addElsetToMesh<T, I, int32_t>(soup, num_elements, dataset, datatype, name);
    } else if (datatype_size == 8) {
      addElsetToMesh<T, I, int64_t>(soup, num_elements, dataset, datatype, name);
    }
  }
}

//==============================================================================
// readXDMFUniformGrid
//==============================================================================

template <std::floating_point T, std::signed_integral I>
  requires(sizeof(T) == 4 || sizeof(T) == 8)
void readXDMFUniformGrid(pugi::xml_node const & xgrid, H5::H5File const & h5file,
                         String const & h5filename, PolytopeSoup<T, I> & mesh)
{
  readXDMFGeometry(xgrid, h5file, h5filename, mesh);
  readXDMFTopology(xgrid, h5file, h5filename, mesh);
  readXDMFElsets(xgrid, h5file, h5filename, mesh);
}

//==============================================================================
// readXDMFFile
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
readXDMFFile(String const & filename, PolytopeSoup<T, I> & soup)
{
  Log::info("Reading XDMF mesh file: " + filename);

  // Open HDF5 file
  Size const h5filepath_end = filename.find_last_of('/') + 1;
  String const h5filename =
      filename.substr(h5filepath_end, filename.size() - 4 - h5filepath_end) + "h5";
  String const h5filepath = filename.substr(0, h5filepath_end);
  LOG_DEBUG("H5 filename: " + h5filename);
  H5::H5File const h5file((h5filepath + h5filename).c_str(), H5F_ACC_RDONLY);

  // Setup XML file
  pugi::xml_document xdoc;
  pugi::xml_parse_result const result = xdoc.load_file(filename.c_str());
  if (!result) {
    Log::error("XDMF XML parse error: " + String(result.description()) +
               ", character pos= " + toString(result.offset));
  }
  pugi::xml_node const xroot = xdoc.child("Xdmf");
  if (strcmp("Xdmf", xroot.name()) != 0) {
    Log::error("XDMF XML root node is not Xdmf");
    return;
  }
  pugi::xml_node const xdomain = xroot.child("Domain");
  if (strcmp("Domain", xdomain.name()) != 0) {
    Log::error("XDMF XML domain node is not Domain");
    return;
  }

  pugi::xml_node const xgrid = xdomain.child("Grid");
  if (strcmp("Grid", xgrid.name()) != 0) {
    Log::error("XDMF XML grid node is not Grid");
    return;
  }
  if (strcmp("Uniform", xgrid.attribute("GridType").value()) == 0) {
    readXDMFUniformGrid(xgrid, h5file, h5filename, soup);
  } else if (strcmp("Tree", xgrid.attribute("GridType").value()) == 0) {
    Log::error("XDMF XML Tree is not supported");
  } else {
    Log::error("XDMF XML grid type is not Uniform or Tree");
  }
}

} // namespace um2
#if defined(__GNUC__) && !defined(__clang__)
#  pragma GCC diagnostic pop
#endif
