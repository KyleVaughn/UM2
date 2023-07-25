#pragma once

#include <um2/common/sto.hpp>
#include <um2/config.hpp>
#include <um2/mesh/MeshFile.hpp>

#include <spdlog/spdlog.h>

#include <charconv>
#include <concepts>
#include <fstream>
#include <string>

namespace um2
{

// -----------------------------------------------------------------------------
// ABAQUS mesh file
// -----------------------------------------------------------------------------
// IO for ABAQUS mesh files.

template <std::floating_point T, std::signed_integral I>
static void
// NOLINTNEXTLINE(misc-unused-parameters)
parseNodes(MeshFile<T, I> & mesh, std::string & line, std::ifstream & file)
{
  // Would love to use chars_format here, but it bugs out on "0.5" occasionally
  SPDLOG_DEBUG("Parsing nodes");
  while (std::getline(file, line) && line[0] != '*') {
    // Format: node_id, x, y, z
    // Skip ID
    size_t last = line.find(',', 0);
    size_t next = line.find(',', last + 2);
    // Read coordinates
    mesh.nodes_x.push_back(sto<T>(line.substr(last + 2, next - last - 2)));
    last = next;
    next = line.find(',', last + 2);
    mesh.nodes_y.push_back(sto<T>(line.substr(last + 2, next - last - 2)));
    mesh.nodes_z.push_back(sto<T>(line.substr(next + 2)));
  }
}

template <std::floating_point T, std::signed_integral I>
static void
// NOLINTNEXTLINE(misc-unused-parameters)
parseElements(MeshFile<T, I> & mesh, std::string & line, std::ifstream & file)
{
  SPDLOG_DEBUG("Parsing elements");
  // "*ELEMENT, type=CPS".size() = 18
  // CPS3 is a 3-node triangle
  // CPS4 is a 4-node quadrilateral
  // CPS6 is a 6-node quadratic triangle
  // CPS8 is a 8-node quadratic quadrilateral
  // Hence, line[18] is the offset of the element type
  // ASCII code for '0' is 48, so line[18] - 48 is the offset
  // as an integer
  I offset = static_cast<I>(line[18]) - 48;
  AbaqusCellType element_type = AbaqusCellType::CPS3;
  switch (offset) {
  case 3:
    element_type = AbaqusCellType::CPS3;
    break;
  case 4:
    element_type = AbaqusCellType::CPS4;
    break;
  case 6:
    element_type = AbaqusCellType::CPS6;
    break;
  case 8:
    element_type = AbaqusCellType::CPS8;
    break;
  default: {
    spdlog::error("AbaqusCellType CPS{} is not supported", offset);
    break;
  }
  }
  // NOLINTBEGIN(cppcoreguidelines-init-variables)
  Size num_elements = 0;
  while (std::getline(file, line) && line[0] != '*') {
    SPDLOG_TRACE("Line: " + line);
    std::string_view line_view = line;
    // For each element, read the element ID and the node IDs
    // Format: id, n1, n2, n3, n4, n5 ...
    // Skip ID
    size_t last = line_view.find(',', 0);
    size_t next = line_view.find(',', last + 2);
    I id = -1;
    while (next != std::string::npos) {
      std::from_chars(line_view.data() + last + 2, line_view.data() + next, id);
      SPDLOG_DEBUG("Node ID: {}", id);
      assert(id > 0);
      mesh.element_conn.push_back(id - 1); // ABAQUS is 1-indexed
      last = next;
      next = line_view.find(',', last + 2);
    }
    // Read last node ID
    std::from_chars(line_view.data() + last + 2, line_view.data() + line_view.size(), id);
    assert(id > 0);
    mesh.element_conn.push_back(id - 1); // ABAQUS is 1-indexed
    num_elements++;
  }
  // NOLINTEND(cppcoreguidelines-init-variables)
  mesh.element_types.push_back(num_elements, static_cast<int8_t>(element_type));
  Size offsets_size = mesh.element_offsets.size();
  if (offsets_size == 0) {
    mesh.element_offsets.push_back(0);
    offsets_size = 1;
  }
  I const offset_back = mesh.element_offsets.back();
  mesh.element_offsets.push_back(num_elements, -1);
  for (Size i = 0; i < num_elements; ++i) {
    I const val = offset_back + static_cast<I>((i + 1) * offset);
    mesh.element_offsets[offsets_size + i] = val;
  }
}

template <std::floating_point T, std::signed_integral I>
static void
parseElsets(MeshFile<T, I> & mesh, std::string & line, std::ifstream & file)
{
  SPDLOG_DEBUG("Parsing elsets");
  std::string_view line_view = line;
  // "*ELSET,ELSET=".size() = 13
  std::string elset_name{line_view.substr(13, line_view.size() - 13)};
  mesh.elset_names.push_back(String(elset_name.c_str()));
  Vector<I> this_elset_ids;
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
    this_elset_ids.push_back(id - 1); // ABAQUS is 1-indexed
    last = next;
    next = line_view.find(',', last + 1);
    while (next != std::string::npos) {
      std::from_chars(line_view.data() + last + 2, line_view.data() + next, id);
      assert(id > 0);
      this_elset_ids.push_back(id - 1); // ABAQUS is 1-indexed
      last = next;
      next = line_view.find(',', last + 1);
    }
  }
  // Ensure the elset is sorted
  assert(std::is_sorted(this_elset_ids.cbegin(), this_elset_ids.cend()));
  mesh.elset_ids.push_back(um2::move(this_elset_ids));
}

template <std::floating_point T, std::signed_integral I>
void
readAbaqusFile(std::string const & filename, MeshFile<T, I> & mesh)
{
  spdlog::info("Reading ABAQUS mesh file: " + filename);

  // Open file
  std::ifstream file(filename);
  if (!file.is_open()) {
    spdlog::error("Could not open file: " + filename);
    return;
  }

  // Set filepath and format
  mesh.filepath = filename;
  mesh.format = MeshFileFormat::Abaqus;

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
  // Sort the elsets (and elset_ids) by name
  Vector<Size> perm;
  sortPermutation(mesh.elset_names, perm);
  applyPermutation(mesh.elset_names, perm);
  applyPermutation(mesh.elset_ids, perm);

  file.close();
} // readAbaqusFile

} // namespace um2
