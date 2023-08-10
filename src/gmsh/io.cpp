#include <um2/gmsh/io.hpp>

#include <um2/common/Color.hpp>
#include <um2/common/Log.hpp>
#include <um2/stdlib/sto.hpp>

#include <algorithm> // std::sort
#include <fstream>   // std::ofstream, std::ifstream
#include <vector>    // std::vector

#if UM2_ENABLE_GMSH

namespace um2::gmsh
{

void
write(std::string const & filename, bool const extra_info)
{
  Log::info("Writing file: " + filename);
  gmsh::write(filename);
  if (!extra_info) {
    return;
  }
  std::string const info_filename =
      filename.substr(0, filename.find_last_of('.')) + ".info";
  std::ofstream info_file(info_filename);
  if (!info_file.is_open()) {
    Log::error("Could not open file " + info_filename);
    return;
  }
  // PHYSICAL_GROUPS
  // ---------------------------------------------------------------------
  gmsh::vectorpair dimtags;
  gmsh::model::getPhysicalGroups(dimtags);
  size_t const num_groups = dimtags.size();
  info_file << "PHYSICAL_GROUPS " << num_groups << std::endl;
  for (auto const & dimtag : dimtags) {
    int const dim = dimtag.first;
    int const tag = dimtag.second;
    std::string name;
    gmsh::model::getPhysicalName(dim, tag, name);
    std::vector<int> tags;
    gmsh::model::getEntitiesForPhysicalGroup(dim, tag, tags);
    size_t const num_tags = tags.size();
    info_file << "PHYSICAL_GROUP \"" << name << "\" " << dim << " " << num_tags << "\n";
    // Write 8 tags per line.
    size_t const num_full_lines = num_tags / 8;
    size_t const num_remaining_tags = num_tags % 8;
    for (size_t i = 0; i < num_full_lines; ++i) {
      info_file << tags[8 * i + 0] << " " << tags[8 * i + 1] << " " << tags[8 * i + 2]
                << " " << tags[8 * i + 3] << " " << tags[8 * i + 4] << " "
                << tags[8 * i + 5] << " " << tags[8 * i + 6] << " " << tags[8 * i + 7]
                << "\n";
    }
    for (size_t i = 0; i + 1 < num_remaining_tags; ++i) {
      info_file << tags[num_full_lines * 8 + i] << " ";
    }
    if (num_remaining_tags > 0) {
      info_file << tags[num_full_lines * 8 + num_remaining_tags - 1] << "\n";
    }
  }
  // ENTITY_COLORS
  // ---------------------------------------------------------------------
  dimtags.clear();
  gmsh::model::getEntities(dimtags);
  std::vector<Color> colors;
  std::vector<gmsh::vectorpair> dimtags_by_color;
  for (auto const & dimtag : dimtags) {
    int const dim = dimtag.first;
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
      colors.push_back(color);
      dimtags_by_color.emplace_back();
      dimtags_by_color.back().push_back(dimtag);
    } else {
      size_t const index = static_cast<size_t>(it - colors.begin());
      dimtags_by_color[index].push_back(dimtag);
    }
  }
  for (auto & color_dimtags : dimtags_by_color) {
    std::sort(color_dimtags.begin(), color_dimtags.end());
  }
  size_t const num_colors = colors.size();
  info_file << "ENTITY_COLORS " << num_colors << std::endl;
  for (size_t i = 0; i < num_colors; ++i) {
    gmsh::vectorpair const & color_dimtags = dimtags_by_color[i];
    size_t const num_dimtags = color_dimtags.size();
    info_file << "ENTITY_COLOR " << static_cast<int>(colors[i].r()) << " "
              << static_cast<int>(colors[i].g()) << " " << static_cast<int>(colors[i].b())
              << " " << static_cast<int>(colors[i].a()) << " " << num_dimtags << "\n";
    size_t const num_full_lines = num_dimtags / 8;
    size_t const num_remaining_dimtags = num_dimtags % 8;
    for (size_t j = 0; j < num_full_lines; ++j) {
      info_file << color_dimtags[8 * j + 0].first << " "
                << color_dimtags[8 * j + 0].second << " "
                << color_dimtags[8 * j + 1].first << " "
                << color_dimtags[8 * j + 1].second << " "
                << color_dimtags[8 * j + 2].first << " "
                << color_dimtags[8 * j + 2].second << " "
                << color_dimtags[8 * j + 3].first << " "
                << color_dimtags[8 * j + 3].second << " "
                << color_dimtags[8 * j + 4].first << " "
                << color_dimtags[8 * j + 4].second << " "
                << color_dimtags[8 * j + 5].first << " "
                << color_dimtags[8 * j + 5].second << " "
                << color_dimtags[8 * j + 6].first << " "
                << color_dimtags[8 * j + 6].second << " "
                << color_dimtags[8 * j + 7].first << " "
                << color_dimtags[8 * j + 7].second << "\n";
    }
    for (size_t j = 0; j + 1 < num_remaining_dimtags; ++j) {
      info_file << color_dimtags[num_full_lines * 8 + j].first << " "
                << color_dimtags[num_full_lines * 8 + j].second << " ";
    }
    if (num_remaining_dimtags > 0) {
      size_t const last = num_full_lines * 8 + num_remaining_dimtags - 1;
      info_file << color_dimtags[last].first << " " << color_dimtags[last].second << "\n";
    }
  }
  // Finished. Close file.
  info_file.close();
}

static void
addPhysicalGroups(std::ifstream & info_file, std::string const & info_filename)
{
  // PHYSICAL_GROUPS
  // ---------------------------------------------------------------------
  std::string line;
  std::getline(info_file, line);
  if (!line.starts_with("PHYSICAL_GROUPS")) {
    Log::error("Could not read PHYSICAL_GROUPS from " + info_filename);
    return;
  }
  size_t const num_groups = sto<size_t>(line.substr(16));
  std::vector<int> tags;
  for (size_t i = 0; i < num_groups; ++i) {
    std::getline(info_file, line);
    if (!line.starts_with("PHYSICAL_GROUP")) {
      Log::error("Could not read PHYSICAL_GROUP from " + info_filename);
      return;
    }
    size_t const name_start = line.find('"') + 1;
    size_t const name_end = line.find('"', name_start);
    std::string const name = line.substr(name_start, name_end - name_start);
    // dim is 1, 2, or 3 and is 2 characters after the end of the name.
    size_t const dim_start = name_end + 2;
    int const dim = sto<int>(line.substr(dim_start, 1));
    assert(dim == 1 || dim == 2 || dim == 3);
    size_t const num_tags_start = dim_start + 2;
    size_t const num_tags = sto<size_t>(line.substr(num_tags_start));
    tags.resize(num_tags);
    size_t const num_full_lines = num_tags / 8;
    size_t const num_remaining_tags = num_tags % 8;
    for (size_t j = 0; j < num_full_lines; ++j) {
      std::getline(info_file, line);
      size_t token_start = 0;
      size_t token_end = line.find(' ');
      for (size_t k = 0; k < 7; ++k) {
        tags[8 * j + k] = sto<int>(line.substr(token_start, token_end - token_start));
        token_start = token_end + 1;
        token_end = line.find(' ', token_start);
      }
      tags[8 * j + 7] = sto<int>(line.substr(token_start));
    }
    if (num_remaining_tags != 0) {
      std::getline(info_file, line);
      size_t token_start = 0;
      for (size_t k = 0; k + 1 < num_remaining_tags; ++k) {
        size_t const token_end = line.find(' ', token_start);
        tags[num_full_lines * 8 + k] =
            sto<int>(line.substr(token_start, token_end - token_start));
        token_start = token_end + 1;
      }
      tags[num_full_lines * 8 + num_remaining_tags - 1] =
          sto<int>(line.substr(token_start));
    }
    gmsh::model::addPhysicalGroup(dim, tags, -1, name);
  }
}

void
open(std::string const & filename, bool const extra_info)
{
  Log::info("Opening file: " + filename);
  // Warn if the file doesn't exist, because Gmsh doesn't...
  {
    std::ifstream const file(filename);
    if (!file.good()) {
      Log::error("File " + filename + " does not exist.");
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
    Log::error("Could not open file " + info_filename);
    return;
  }
  // PHYSICAL_GROUPS
  // ---------------------------------------------------------------------
  addPhysicalGroups(info_file, info_filename);

  // ENTITY_COLORS
  // ---------------------------------------------------------------------
  {
    std::string line;
    std::getline(info_file, line);
    if (!line.starts_with("ENTITY_COLORS")) {
      Log::error("Could not read ENTITY_COLORS from " + info_filename);
      return;
    }
    size_t const num_colors = sto<size_t>(line.substr(14));
    for (size_t i = 0; i < num_colors; ++i) {
      std::getline(info_file, line);
      if (!line.starts_with("ENTITY_COLOR")) {
        Log::error("Could not read ENTITY_COLOR from " + info_filename);
        return;
      }
      size_t token_start = 13;
      size_t token_end = line.find(' ', token_start);
      int const r = sto<int>(line.substr(token_start, token_end - token_start));
      token_start = token_end + 1;
      token_end = line.find(' ', token_start);
      int const g = sto<int>(line.substr(token_start, token_end - token_start));
      token_start = token_end + 1;
      token_end = line.find(' ', token_start);
      int const b = sto<int>(line.substr(token_start, token_end - token_start));
      token_start = token_end + 1;
      token_end = line.find(' ', token_start);
      int const a = sto<int>(line.substr(token_start, token_end - token_start));
      token_start = token_end + 1;
      size_t const num_dimtags = sto<size_t>(line.substr(token_start));
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
              sto<int>(line.substr(token_start, token_end - token_start));
          token_start = token_end + 1;
          // Tag
          token_end = line.find(' ', token_start);
          dimtags[8 * j + k].second =
              sto<int>(line.substr(token_start, token_end - token_start));
          token_start = token_end + 1;
        }
        token_end = line.find(' ', token_start);
        dimtags[8 * j + 7].first =
            sto<int>(line.substr(token_start, token_end - token_start));
        token_start = token_end + 1;
        dimtags[8 * j + 7].second = sto<int>(line.substr(token_start));
      }
      if (num_remaining_dimtags != 0) {
        std::getline(info_file, line);
        token_start = 0;
        for (size_t k = 0; k + 1 < num_remaining_dimtags; ++k) {
          // Dim
          token_end = line.find(' ', token_start);
          dimtags[num_full_lines * 8 + k].first =
              sto<int>(line.substr(token_start, token_end - token_start));
          token_start = token_end + 1;
          // Tag
          token_end = line.find(' ', token_start);
          dimtags[num_full_lines * 8 + k].second =
              sto<int>(line.substr(token_start, token_end - token_start));
          token_start = token_end + 1;
        }
        token_end = line.find(' ', token_start);
        dimtags[num_full_lines * 8 + num_remaining_dimtags - 1].first =
            sto<int>(line.substr(token_start, token_end - token_start));
        token_start = token_end + 1;
        dimtags[num_full_lines * 8 + num_remaining_dimtags - 1].second =
            sto<int>(line.substr(token_start));
      }
      gmsh::model::setColor(dimtags, r, g, b, a);
    }
  }

  // Finished reading info file
  info_file.close();
}

} // namespace um2::gmsh
#endif // UM2_ENABLE_GMSH
