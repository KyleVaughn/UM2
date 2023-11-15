#pragma once

#include <um2/config.hpp>

#include <charconv>
#include <string>
#include <vector>

namespace um2
{

// Convert a single or multi-line string to a vector of vectors of type T.
// Ex: to_vecvec<int>(R"1 2 3
//                      4 5 6"); == {{1, 2, 3}, {4, 5, 6}}
//
// Primarily used for MPACT input related functions, e.g. getting the pin IDs
// in a ray tracing module.

template <typename T>
constexpr auto
// We have a class named Vec, so we need to use a different name for this function.
// NOLINTNEXTLINE(readability-identifier-naming) justified
to_vecvec(std::string const & str, std::string const & delimiter = " ")
    -> std::vector<std::vector<T>>
{
  size_t const delim_size = delimiter.size();
  std::vector<std::vector<T>> result;
  // Loop through all newline characters, handling the string on a line-by-line basis.
  // If the string ends with a newline, this loop will handle the last line.
  // Otherwise, the last line will be handled after the loop.
  size_t line_start = 0;
  size_t line_stop = 0;
  std::string_view const str_view(str);
  std::string_view line;
  while ((line_stop = str_view.find('\n', line_start)) != std::string::npos) {
    line = str_view.substr(line_start, line_stop - line_start);
    line_start = line_stop + 1;
    // If the line is non-empty and contains non-whitespace characters, create a new row.
    size_t token_start = line.find_first_not_of(' ');
    if (token_start != std::string::npos) {
      std::vector<T> line_vec;
      size_t token_stop = line.find(delimiter, token_start);
      T value;
      while (token_stop != std::string::npos) {
        std::from_chars(line.data() + token_start, line.data() + token_stop, value);
        line_vec.push_back(value);
        token_start = token_stop + delim_size;
        token_stop = line.find(delimiter, token_start);
      }
      std::from_chars(line.data() + token_start, line.data() + line.size(), value);
      line_vec.push_back(value);
      result.push_back(line_vec);
    }
  }
  // Check to see if the string ends in a newline. If it does, the last line has
  // already been handled. Otherwise, handle the last line.
  if (line_start < str.size()) {
    line = str_view.substr(line_start, line_stop - line_start);
    size_t token_start = line.find_first_not_of(' ');
    if (token_start != std::string::npos) {
      std::vector<T> line_vec;
      size_t token_stop = line.find(delimiter, token_start);
      // If the line contains no delimiter, the line is a single token.
      // Otherwise, the line contains multiple tokens.
      T value;
      while (token_stop != std::string::npos) {
        std::from_chars(line.data() + token_start, line.data() + token_stop, value);
        line_vec.push_back(value);
        token_start = token_stop + delim_size;
        token_stop = line.find(delimiter, token_start);
      }
      std::from_chars(line.data() + token_start, line.data() + line.size(), value);
      line_vec.push_back(value);
      result.push_back(line_vec);
    }
  }
  return result;
}

} // namespace um2
