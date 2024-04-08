#pragma once

#include <um2/stdlib/vector.hpp>
#include <um2/stdlib/string.hpp>
#include <um2/common/strto.hpp>

namespace um2
{

// Convert a single or multi-line string to a vector of vectors of type T.
// Ex: to_vecvec<int>(R"1 2 3
//                      4 5 6"); == {{1, 2, 3}, {4, 5, 6}}

template <class T>
auto
stringToLattice(String const & str, char const delimiter = ' ')
    -> Vector<Vector<T>>
{
  Vector<Vector<T>> result;

  // Remove leading spaces
  StringView str_view(str);
  str_view.removeLeadingSpaces();

  // For each new line
  StringView line = str_view.getTokenAndShrink('\n');
  line.removeLeadingSpaces();
  while (!line.empty()) {
    Vector<T> line_vec;

    // For each token in the line
    StringView token = line.getTokenAndShrink(delimiter);
    token.removeLeadingSpaces();
    while (!token.empty()) {
      char * end = nullptr;
      line_vec.emplace_back(strto<T>(token.data(), &end));
      ASSERT(end != nullptr);
      token = line.getTokenAndShrink(delimiter);
      token.removeLeadingSpaces();
    }
    // Add the line to the result
    result.emplace_back(um2::move(line_vec));

    // Get the next line
    line = str_view.getTokenAndShrink('\n');
    line.removeLeadingSpaces();
  }

#if UM2_ENABLE_ASSERTS
  str_view.removeLeadingSpaces();
  bool const last_line_empty = str_view.empty();
  bool const last_line_has_more_values = str_view.find_first_not_of(delimiter) == StringView::npos;
  ASSERT(last_line_empty || last_line_has_more_values);
#endif

  return result;
}

} // namespace um2
