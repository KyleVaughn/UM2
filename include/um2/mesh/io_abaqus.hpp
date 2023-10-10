#pragma once

#include <um2/config.hpp>

#include <um2/common/Log.hpp>
#include <um2/mesh/MeshFile.hpp>
#include <um2/stdlib/sto.hpp>

#include <charconv>
#include <concepts>
#include <fstream>
#include <string>

namespace um2
{

//==============================================================================-
// ABAQUS mesh file
//==============================================================================
//
// IO for ABAQUS mesh files.

//==============================================================================-
// parseNodes
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
parseNodes(MeshFile<T, I> & mesh, std::string & line, std::ifstream & file)
{
  // Would love to use chars_format here, but it bugs out on "0.5" occasionally
  LOG_DEBUG("Parsing nodes");
  while (std::getline(file, line) && line[0] != '*') {
    // Format: node_id, x, y, z
    // Skip ID
    size_t last = line.find(',', 0);
    size_t next = line.find(',', last + 2);
    // Read coordinates
    T const x = sto<T>(line.substr(last + 2, next - last - 2));
    last = next;
    next = line.find(',', last + 2);
    T const y = sto<T>(line.substr(last + 2, next - last - 2));
    T const z = sto<T>(line.substr(next + 2));
    mesh.vertices.emplace_back(x, y, z);
  }
}

//==============================================================================-
// parseElements
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
parseElements(MeshFile<T, I> & mesh, std::string & line, std::ifstream & file)
{
  LOG_DEBUG("Parsing elements");
  //  "*ELEMENT, type=CPS".size() = 18
  //  CPS3 is a 3-node triangle
  //  CPS4 is a 4-node quadrilateral
  //  CPS6 is a 6-node quadratic triangle
  //  CPS8 is a 8-node quadratic quadrilateral
  //  Hence, line[18] is the offset of the element type
  //  ASCII code for '0' is 48, so line[18] - 48 is the offset
  //  as an integer
  //
  assert(line[15] == 'C' && line[16] == 'P' && line[17] == 'S');
  I const offset = static_cast<I>(line[18]) - 48;
  MeshType this_type = MeshType::None;
  switch (offset) {
  case 3:
    this_type = MeshType::Tri;
    break;
  case 4:
    this_type = MeshType::Quad;
    break;
  case 6:
    this_type = MeshType::QuadraticTri;
    break;
  case 8:
    this_type = MeshType::QuadraticQuad;
    break;
  default: {
    LOG_ERROR("AbaqusCellType CPS" + toString(offset) + " is not supported");
    break;
  }
  }
  size_t num_elements = 0;
  while (std::getline(file, line) && line[0] != '*') {
    LOG_TRACE("Line: " + String(line.c_str()));
    std::string_view const line_view = line;
    // For each element, read the element ID and the node IDs
    // Format: id, n1, n2, n3, n4, n5 ...
    // Skip ID
    size_t last = line_view.find(',', 0);
    size_t next = line_view.find(',', last + 2);
    I id = -1;
    while (next != std::string::npos) {
      std::from_chars(line_view.data() + last + 2, line_view.data() + next, id);
      LOG_TRACE("Node ID: " + toString(id));
      assert(id > 0);
      mesh.element_conn.push_back(id - 1); // ABAQUS is 1-indexed
      last = next;
      next = line_view.find(',', last + 2);
    }
    // Read last node ID
    std::from_chars(line_view.data() + last + 2, line_view.data() + line_view.size(), id);
    assert(id > 0);
    mesh.element_conn.push_back(id - 1); // ABAQUS is 1-indexed
    ++num_elements;
  }
  mesh.element_types.insert(mesh.element_types.end(), num_elements, this_type);
  size_t offsets_size = mesh.element_offsets.size();
  if (offsets_size == 0) {
    mesh.element_offsets.push_back(0);
    offsets_size = 1;
  }
  I const offset_back = mesh.element_offsets.back();
  mesh.element_offsets.resize(offsets_size + num_elements);
  for (size_t i = 0; i < num_elements; ++i) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
    auto const ip1 = static_cast<I>(i + 1U);
    mesh.element_offsets[offsets_size + i] = offset_back + ip1 * offset;
#pragma GCC diagnostic pop
  }
}

//==============================================================================
// parseElsets
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
parseElsets(MeshFile<T, I> & mesh, std::string & line, std::ifstream & file)
{
  LOG_DEBUG("Parsing elsets");
  std::string_view line_view = line;
  // "*ELSET,ELSET=".size() = 13
  std::string const elset_name{line_view.substr(13, line_view.size() - 13)};
  mesh.elset_names.emplace_back(elset_name);
  if (mesh.elset_offsets.size() == 0) {
    mesh.elset_offsets.push_back(0);
  }
  I const offset_back = mesh.elset_offsets.back();
  I num_elements = 0;
  while (std::getline(file, line) && line[0] != '*') {
    line_view = line;
    // Add each element ID to the elset
    // Format: id, id, id, id, id,
    // Note, line ends in ", " or ","
    // First ID
    size_t last = 0;
    size_t next = line_view.find(',');
    I id;
    std::from_chars(line_view.data(), line_view.data() + next, id);
    assert(id > 0);
    mesh.elset_ids.push_back(id - 1); // ABAQUS is 1-indexed
    num_elements++;
    last = next;
    next = line_view.find(',', last + 1);
    while (next != std::string::npos) {
      std::from_chars(line_view.data() + last + 2, line_view.data() + next, id);
      assert(id > 0);
      mesh.elset_ids.push_back(id - 1); // ABAQUS is 1-indexed
      num_elements++;
      last = next;
      next = line_view.find(',', last + 1);
    }
  }
  mesh.elset_offsets.push_back(offset_back + num_elements);
  // Ensure the elset is sorted
  assert(std::is_sorted(mesh.elset_ids.cbegin() + offset_back, mesh.elset_ids.cend()));
}

//==============================================================================
// readAbaqusFile
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
readAbaqusFile(std::string const & filename, MeshFile<T, I> & mesh)
{
  LOG_INFO("Reading Abaqus mesh file: " + String(filename.c_str()));

  // Open file
  std::ifstream file(filename);
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: " + String(filename.c_str()));
    return;
  }

  // Set filepath
  mesh.filepath = filename;

  // Read file
  std::string line;
  bool loop_again = false;
  while (loop_again || std::getline(file, line)) {
    loop_again = false;
    if (line.starts_with("*Heading")) {
      std::getline(file, line);
      size_t nchars = line.size() - 1; // Omit leading space
      // If name ends in .inp, remove it
      if (line.ends_with(".inp")) {
        nchars -= 4;
      }
      mesh.name = line.substr(1, nchars);
    } else if (line.starts_with("*NODE")) {
      parseNodes<T>(mesh, line, file);
      loop_again = true;
    } else if (line.starts_with("*ELEMENT")) {
      parseElements<T, I>(mesh, line, file);
      loop_again = true;
    } else if (line.starts_with("*ELSET")) {
      parseElsets<T, I>(mesh, line, file);
      loop_again = true;
    }
  }
  mesh.sortElsets();
  file.close();
} // readAbaqusFile

} // namespace um2
