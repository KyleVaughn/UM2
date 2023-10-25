#pragma once

#include <um2/config.hpp>

#include <um2/common/Log.hpp>
#include <um2/mesh/PolytopeSoup.hpp>
#include <um2/stdlib/sto.hpp>

#include <charconv>
#include <concepts>
#include <fstream>
#include <string>

namespace um2
{

//==============================================================================-
// IO for ABAQUS files.
//==============================================================================

//==============================================================================-
// parseNodes
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
parseNodes(PolytopeSoup<T, I> & soup, std::string & line, std::ifstream & file)
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
    soup.vertices.emplace_back(x, y, z);
  }
}

//==============================================================================-
// parseElements
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
parseElements(PolytopeSoup<T, I> & soup, std::string & line, std::ifstream & file)
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
  VTKElemType this_type = VTKElemType::Vertex;
  switch (offset) {
  case 3:
    this_type = VTKElemType::Triangle;
    break;
  case 4:
    this_type = VTKElemType::Quad;
    break;
  case 6:
    this_type = VTKElemType::QuadraticTriangle;
    break;
  case 8:
    this_type = VTKElemType::QuadraticQuad;
    break;
  default: {
    LOG_ERROR("AbaqusCellType CPS" + toString(offset) + " is not supported");
    break;
  }
  }
  Size num_elements = 0;
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
      soup.element_conn.push_back(id - 1); // ABAQUS is 1-indexed
      last = next;
      next = line_view.find(',', last + 2);
    }
    // Read last node ID
    std::from_chars(line_view.data() + last + 2, line_view.data() + line_view.size(), id);
    assert(id > 0);
    soup.element_conn.push_back(id - 1); // ABAQUS is 1-indexed
    ++num_elements;
  }
  soup.element_types.push_back(num_elements, this_type);
  Size offsets_size = soup.element_offsets.size();
  if (offsets_size == 0) {
    soup.element_offsets.push_back(0);
    offsets_size = 1;
  }
  I const offset_back = soup.element_offsets.back();
  soup.element_offsets.resize(offsets_size + num_elements);
  for (Size i = 0; i < num_elements; ++i) {
    // offsets_size and i are of type Size
    // offset_back, ip1, and ofset are of type I.
    // How does FMA of I change its type to Size?
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wconversion"
    auto const ip1 = static_cast<I>(i + 1);
    soup.element_offsets[offsets_size + i] = offset_back + ip1 * offset;
    #pragma GCC diagnostic pop
  }
}

//==============================================================================
// parseElsets
//==============================================================================

template <std::floating_point T, std::signed_integral I>
static void
parseElsets(PolytopeSoup<T, I> & soup, std::string & line, std::ifstream & file)
{
  LOG_DEBUG("Parsing elsets");
  std::string_view line_view = line;
  // "*ELSET,ELSET=".size() = 13
  std::string const elset_name_std{line_view.substr(13, line_view.size() - 13)};
  String const elset_name(elset_name_std.c_str());
  Vector<I> elset_ids;
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
    elset_ids.push_back(id - 1); // ABAQUS is 1-indexed
    last = next;
    next = line_view.find(',', last + 1);
    while (next != std::string::npos) {
      std::from_chars(line_view.data() + last + 2, line_view.data() + next, id);
      assert(id > 0);
      elset_ids.push_back(id - 1); // ABAQUS is 1-indexed
      last = next;
      next = line_view.find(',', last + 1);
    }
  }
  soup.addElset(elset_name, elset_ids);
  // Ensure the elset is sorted
  assert(um2::is_sorted(elset_ids.cbegin(), elset_ids.cend()));
}

//==============================================================================
// readAbaqusFile
//==============================================================================

template <std::floating_point T, std::signed_integral I>
void
readAbaqusFile(String const & filename, PolytopeSoup<T, I> & soup)
{
  LOG_INFO("Reading Abaqus file: " + filename);

  // Open file
  std::ifstream file(filename.c_str());
  if (!file.is_open()) {
    LOG_ERROR("Could not open file: " + filename);
    return;
  }

  // Read file
  std::string line;
  bool loop_again = false;
  while (loop_again || std::getline(file, line)) {
    loop_again = false;
    if (line.starts_with("*NODE")) {
      parseNodes<T>(soup, line, file);
      loop_again = true;
    } else if (line.starts_with("*ELEMENT")) {
      parseElements<T, I>(soup, line, file);
      loop_again = true;
    } else if (line.starts_with("*ELSET")) {
      parseElsets<T, I>(soup, line, file);
      loop_again = true;
    }
  }
  soup.sortElsets();
  file.close();
  LOG_INFO("Finished reading Abaqus file: " + filename);
} // readAbaqusFile

} // namespace um2
