#pragma once

#include <um2/common/bit_ceil.hpp>
#include <um2/common/config.hpp>

#include <cstring>
//#include <ostream>
#include <string>

namespace um2
{

// -----------------------------------------------------------------------------
// STRING
// -----------------------------------------------------------------------------
// A std::string-like class, but without an allocator template parameter.
// Allocates 2^N elements, where N is the smallest integer such that 2^N >= size.
// Use this in place of std::string for hostdev/device code.
//
// Stores a null-terminator at the end of the string.

struct String {

private:
  len_t _size = 0;
  len_t _capacity = 0;
  char8_t * _data = nullptr;

public:
  // -- Destructor --

  UM2_HOSTDEV ~String() { delete[] _data; }

  // -- Accessors --

  UM2_PURE UM2_HOSTDEV constexpr char8_t * begin() const;

  UM2_PURE UM2_HOSTDEV constexpr char8_t * end() const;

  UM2_PURE UM2_HOSTDEV constexpr char8_t const * cbegin() const;

  UM2_PURE UM2_HOSTDEV constexpr char8_t const * cend() const;

  UM2_PURE UM2_HOSTDEV constexpr len_t size() const;

  UM2_PURE UM2_HOSTDEV constexpr len_t capacity() const;

  UM2_PURE UM2_HOSTDEV constexpr char8_t * data();

  UM2_PURE UM2_HOSTDEV constexpr char8_t const * data() const;

  // -- Constructors --

  constexpr String() = default;

  template <size_t N>
  UM2_HOSTDEV explicit String(char const (&)[N]);

  UM2_HOSTDEV String(String const &);

  explicit String(std::string const &);

  // -- Operators --

  UM2_HOSTDEV String & operator=(String const &);

  template <size_t N>
  UM2_HOSTDEV String & operator=(char const (&)[N]);

  String & operator=(std::string const &);

  UM2_PURE UM2_HOSTDEV constexpr bool operator==(String const &) const;

  template <size_t N>
  UM2_PURE UM2_HOSTDEV constexpr bool operator==(char const (&)[N]) const;

  UM2_PURE constexpr bool operator==(std::string const &) const;

  UM2_PURE UM2_HOSTDEV bool operator<(String const &) const;

  UM2_PURE UM2_HOSTDEV bool operator>(String const &) const;

  UM2_PURE UM2_HOSTDEV bool operator<=(String const &) const;

  UM2_PURE UM2_HOSTDEV bool operator>=(String const &) const;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr char8_t & operator[](len_t);

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr char8_t const & operator[](len_t) const;

  // -- Methods --

  UM2_PURE UM2_HOSTDEV int compare(String const &) const;

  //    UM2_PURE UM2_HOSTDEV constexpr bool contains(char const) const;
  //
  //    UM2_PURE constexpr bool starts_with(std::string const &) const;
  //
  //    UM2_PURE constexpr bool ends_with(std::string const &) const;
  //

}; // struct String

// // -- Methods --
//
// UM2_PURE std::string to_string(String const & v);
//
// // -- IO --
//
// std::ostream & operator << (std::ostream &, String const &);

} // namespace um2

#include "string.inl"
