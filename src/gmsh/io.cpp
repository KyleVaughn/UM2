#include <um2/config.hpp>

// Personally, I despise using std:string and std::getline for file I/O due to
// the potential for memory allocation and deallocation. I prefer to use C-style
// file I/O, but we only expect these routines to be called once, and only for
// a few thousand lines at most, so it's not worth the refactoring effort.
// - Kyle

#if UM2_USE_GMSH

#  include <um2/common/color.hpp>
#  include <um2/common/logger.hpp>
#  include <um2/gmsh/base_gmsh_api.hpp>
#  include <um2/gmsh/io.hpp>
#  include <um2/stdlib/assert.hpp>

#  include <algorithm>
#  include <cstddef>
#  include <fstream>
#  include <iterator>
#  include <string>
#  include <vector>

namespace um2::gmsh
{

//=============================================================================
// write
//=============================================================================

void
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
write(std::string const & filename, bool const extra_info)
{
  LOG_INFO("Writing file: ", filename.c_str());
  gmsh::write(filename);
  if (!extra_info) {
    return;
  }
  std::string const info_filename =
      filename.substr(0, filename.find_last_of('.')) + ".info";
  std::ofstream info_file(info_filename);
  if (!info_file.is_open()) {
    LOG_ERROR("Could not open file ", info_filename.c_str());
    return;
  }

  // We have to map the entites to a continuous range of tags
  std::vector<std::vector<int>> tag_map(4);

  gmsh::vectorpair dimtags;
  gmsh::model::getEntities(dimtags);
  for (auto const & dimtag : dimtags) {
    auto const dim = static_cast<size_t>(dimtag.first);
    tag_map[dim].emplace_back(dimtag.second);
  }

  for (size_t dim = 0; dim < 4; ++dim) {
    // Should be sorted, but just in case...
    std::sort(tag_map[dim].begin(), tag_map[dim].end());
  }
  dimtags.clear();

  //==============================================================================
  // PHYSICAL_GROUPS
  //==============================================================================
  gmsh::model::getPhysicalGroups(dimtags);
  size_t const num_groups = dimtags.size();
  info_file << "PHYSICAL_GROUPS " << num_groups << '\n';
  for (auto const & dimtag : dimtags) {
    auto const idim = dimtag.first;
    auto const dim = static_cast<size_t>(dimtag.first);
    int const tag = dimtag.second;
    std::string name;
    gmsh::model::getPhysicalName(idim, tag, name);
    std::vector<int> tags;
    gmsh::model::getEntitiesForPhysicalGroup(idim, tag, tags);
    size_t const num_tags = tags.size();
    info_file << "PHYSICAL_GROUP \"" << name << "\" " << dim << ' ' << num_tags << '\n';
    // Write 8 tags per line.
    size_t const num_full_lines = num_tags / 8;
    size_t const num_remaining_tags = num_tags % 8;
    for (size_t i = 0; i < num_full_lines; ++i) {
      for (size_t j = 0; j < 8; ++j) {
        // Find the new tag (index into tag_map + 1)
        auto const it =
            std::lower_bound(tag_map[dim].begin(), tag_map[dim].end(), tags[8 * i + j]);
        // Ensure the tag was found
        ASSERT(it != tag_map[dim].end());
        ASSERT(*it == tags[8 * i + j]);
        info_file << std::distance(tag_map[dim].begin(), it) + 1 << ' ';
      }
      info_file << '\n';
    }
    for (size_t i = 0; i + 1 < num_remaining_tags; ++i) {
      auto const it = std::lower_bound(tag_map[dim].begin(), tag_map[dim].end(),
                                       tags[num_full_lines * 8 + i]);
      ASSERT(it != tag_map[dim].end());
      ASSERT(*it == tags[num_full_lines * 8 + i]);
      info_file << std::distance(tag_map[dim].begin(), it) + 1 << ' ';
    }
    if (num_remaining_tags > 0) {
      auto const it = std::lower_bound(tag_map[dim].begin(), tag_map[dim].end(),
                                       tags[num_full_lines * 8 + num_remaining_tags - 1]);
      ASSERT(it != tag_map[dim].end());
      ASSERT(*it == tags[num_full_lines * 8 + num_remaining_tags - 1]);
      info_file << std::distance(tag_map[dim].begin(), it) + 1 << '\n';
    }
  }
  //==============================================================================
  // ENTITY_COLORS
  //==============================================================================
  dimtags.clear();
  gmsh::model::getEntities(dimtags);
  std::vector<Color> colors;
  std::vector<gmsh::vectorpair> dimtags_by_color;
  for (auto const & dimtag : dimtags) {
    int const dim = dimtag.first;
    if (dim < 2) {
      continue;
    }
    int const tag = dimtag.second;
    int r = 0;
    int g = 0;
    int b = 0;
    int a = 0;
    gmsh::model::getColor(dim, tag, r, g, b, a);
    // Only track colors if they are not the default.
    if (r == 0 && g == 0 && b == 255 && a == 0) {
      continue;
    }
    Color const color(r, g, b, a);
    auto it = std::find(colors.begin(), colors.end(), color);
    if (it == colors.end()) {
      colors.emplace_back(color);
      dimtags_by_color.emplace_back();
      dimtags_by_color.back().emplace_back(dimtag);
    } else {
      size_t const index = static_cast<size_t>(it - colors.begin());
      dimtags_by_color[index].emplace_back(dimtag);
    }
  }
  for (auto & color_dimtags : dimtags_by_color) {
    std::sort(color_dimtags.begin(), color_dimtags.end());
  }
  size_t const num_colors = colors.size();
  info_file << "ENTITY_COLORS " << num_colors << '\n';
  for (size_t i = 0; i < num_colors; ++i) {
    gmsh::vectorpair const & color_dimtags = dimtags_by_color[i];
    size_t const num_dimtags = color_dimtags.size();
    info_file << "ENTITY_COLOR " << static_cast<int>(colors[i].r()) << ' '
              << static_cast<int>(colors[i].g()) << ' ' << static_cast<int>(colors[i].b())
              << ' ' << static_cast<int>(colors[i].a()) << ' ' << num_dimtags << '\n';
    size_t const num_full_lines = num_dimtags / 8;
    size_t const num_remaining_dimtags = num_dimtags % 8;
    for (size_t j = 0; j < num_full_lines; ++j) {
      for (size_t k = 0; k < 8; ++k) {
        auto const cdim = static_cast<size_t>(color_dimtags[8 * j + k].first);
        auto const ctag = color_dimtags[8 * j + k].second;
        auto const it =
            std::lower_bound(tag_map[cdim].begin(), tag_map[cdim].end(), ctag);
        ASSERT(it != tag_map[cdim].end());
        ASSERT(*it == ctag);
        info_file << cdim << ' ' << std::distance(tag_map[cdim].begin(), it) + 1 << ' ';
      }
      info_file << '\n';
    }
    for (size_t j = 0; j + 1 < num_remaining_dimtags; ++j) {
      auto const cdim = static_cast<size_t>(color_dimtags[num_full_lines * 8 + j].first);
      auto const ctag = color_dimtags[num_full_lines * 8 + j].second;
      auto const it = std::lower_bound(tag_map[cdim].begin(), tag_map[cdim].end(), ctag);
      ASSERT(it != tag_map[cdim].end());
      ASSERT(*it == ctag);
      info_file << cdim << ' ' << std::distance(tag_map[cdim].begin(), it) + 1 << ' ';
    }
    if (num_remaining_dimtags > 0) {
      size_t const last = num_full_lines * 8 + num_remaining_dimtags - 1;
      auto const cdim = static_cast<size_t>(color_dimtags[last].first);
      auto const ctag = color_dimtags[last].second;
      auto const it = std::lower_bound(tag_map[cdim].begin(), tag_map[cdim].end(), ctag);
      ASSERT(it != tag_map[cdim].end());
      ASSERT(*it == ctag);
      info_file << cdim << ' ' << std::distance(tag_map[cdim].begin(), it) + 1 << '\n';
    }
  }
  // Finished. Close file.
  info_file.close();
} // write

namespace
{
void
addPhysicalGroups(std::ifstream & info_file, std::string const & info_filename)
{
  //==============================================================================
  // PHYSICAL_GROUPS
  //==============================================================================
  std::string line;
  std::getline(info_file, line);
  if (!line.starts_with("PHYSICAL_GROUPS")) {
    LOG_ERROR("Could not read PHYSICAL_GROUPS from ", info_filename.c_str());
    return;
  }
  size_t const num_groups = std::stoul(line.substr(16));
  std::vector<int> tags;
  for (size_t i = 0; i < num_groups; ++i) {
    std::getline(info_file, line);
    if (!line.starts_with("PHYSICAL_GROUP")) {
      LOG_ERROR("Could not read PHYSICAL_GROUP from ", info_filename.c_str());
      return;
    }
    size_t const name_start = line.find('"') + 1;
    size_t const name_end = line.find('"', name_start);
    std::string const name = line.substr(name_start, name_end - name_start);
    // dim is 1, 2, or 3 and is 2 characters after the end of the name.
    size_t const dim_start = name_end + 2;
    int const dim = std::stoi(line.substr(dim_start, 1));
    ASSERT(dim > 0);
    ASSERT(dim < 4);

    size_t const num_tags_start = dim_start + 2;
    size_t const num_tags = std::stoul(line.substr(num_tags_start));
    tags.resize(num_tags);
    size_t const num_full_lines = num_tags / 8;
    size_t const num_remaining_tags = num_tags % 8;
    for (size_t j = 0; j < num_full_lines; ++j) {
      std::getline(info_file, line);
      size_t token_start = 0;
      size_t token_end = line.find(' ');
      for (size_t k = 0; k < 7; ++k) {
        tags[8 * j + k] = std::stoi(line.substr(token_start, token_end - token_start));
        token_start = token_end + 1;
        token_end = line.find(' ', token_start);
      }
      tags[8 * j + 7] = std::stoi(line.substr(token_start));
    }
    if (num_remaining_tags != 0) {
      std::getline(info_file, line);
      size_t token_start = 0;
      for (size_t k = 0; k + 1 < num_remaining_tags; ++k) {
        size_t const token_end = line.find(' ', token_start);
        tags[num_full_lines * 8 + k] =
            std::stoi(line.substr(token_start, token_end - token_start));
        token_start = token_end + 1;
      }
      tags[num_full_lines * 8 + num_remaining_tags - 1] =
          std::stoi(line.substr(token_start));
    }
    gmsh::model::addPhysicalGroup(dim, tags, -1, name);
  }
}
} // namespace

//==============================================================================
// open
//==============================================================================

void
open(std::string const & filename, bool const extra_info)
{
  LOG_INFO("Opening file: ", filename.c_str());
  // Warn if the file doesn't exist, because Gmsh doesn't...
  {
    std::ifstream const file(filename);
    if (!file.good()) {
      LOG_ERROR("File ", filename.c_str(), " does not exist.");
      return;
    }
  }
  gmsh::open(filename);
  if (!extra_info) {
    return;
  }
  // Read info file.
  std::string const info_filename =
      filename.substr(0, filename.find_last_of('.')) + ".info";
  std::ifstream info_file(info_filename);
  if (!info_file.is_open()) {
    LOG_ERROR("Could not open file ", info_filename.c_str());
    return;
  }
  //==============================================================================
  // PHYSICAL_GROUPS
  //==============================================================================
  addPhysicalGroups(info_file, info_filename);

  //==============================================================================
  // ENTITY_COLORS
  //==============================================================================
  {
    std::string line;
    std::getline(info_file, line);
    if (!line.starts_with("ENTITY_COLORS")) {
      LOG_ERROR("Could not read ENTITY_COLORS from ", info_filename.c_str());
      return;
    }
    size_t const num_colors = std::stoul(line.substr(14));
    for (size_t i = 0; i < num_colors; ++i) {
      std::getline(info_file, line);
      if (!line.starts_with("ENTITY_COLOR")) {
        LOG_ERROR("Could not read ENTITY_COLOR from ", info_filename.c_str());
        return;
      }
      size_t token_start = 13;
      size_t token_end = line.find(' ', token_start);
      int const r = std::stoi(line.substr(token_start, token_end - token_start));
      token_start = token_end + 1;
      token_end = line.find(' ', token_start);
      int const g = std::stoi(line.substr(token_start, token_end - token_start));
      token_start = token_end + 1;
      token_end = line.find(' ', token_start);
      int const b = std::stoi(line.substr(token_start, token_end - token_start));
      token_start = token_end + 1;
      token_end = line.find(' ', token_start);
      int const a = std::stoi(line.substr(token_start, token_end - token_start));
      token_start = token_end + 1;
      size_t const num_dimtags = std::stoul(line.substr(token_start));
      gmsh::vectorpair dimtags(num_dimtags);
      size_t const num_full_lines = num_dimtags / 8;
      size_t const num_remaining_dimtags = num_dimtags % 8;
      for (size_t j = 0; j < num_full_lines; ++j) {
        std::getline(info_file, line);
        token_start = 0;
        for (size_t k = 0; k < 7; ++k) {
          // Dim
          token_end = line.find(' ', token_start);
          dimtags[8 * j + k].first =
              std::stoi(line.substr(token_start, token_end - token_start));
          token_start = token_end + 1;
          // Tag
          token_end = line.find(' ', token_start);
          dimtags[8 * j + k].second =
              std::stoi(line.substr(token_start, token_end - token_start));
          token_start = token_end + 1;
        }
        token_end = line.find(' ', token_start);
        dimtags[8 * j + 7].first =
            std::stoi(line.substr(token_start, token_end - token_start));
        token_start = token_end + 1;
        dimtags[8 * j + 7].second = std::stoi(line.substr(token_start));
      }
      if (num_remaining_dimtags != 0) {
        std::getline(info_file, line);
        token_start = 0;
        for (size_t k = 0; k + 1 < num_remaining_dimtags; ++k) {
          // Dim
          token_end = line.find(' ', token_start);
          dimtags[num_full_lines * 8 + k].first =
              std::stoi(line.substr(token_start, token_end - token_start));
          token_start = token_end + 1;
          // Tag
          token_end = line.find(' ', token_start);
          dimtags[num_full_lines * 8 + k].second =
              std::stoi(line.substr(token_start, token_end - token_start));
          token_start = token_end + 1;
        }
        token_end = line.find(' ', token_start);
        dimtags[num_full_lines * 8 + num_remaining_dimtags - 1].first =
            std::stoi(line.substr(token_start, token_end - token_start));
        token_start = token_end + 1;
        dimtags[num_full_lines * 8 + num_remaining_dimtags - 1].second =
            std::stoi(line.substr(token_start));
      }
      gmsh::model::setColor(dimtags, r, g, b, a);
    }
  }

  // Finished reading info file
  info_file.close();
}

} // namespace um2::gmsh
#endif // UM2_USE_GMSH
