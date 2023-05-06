#pragma once

#include <um2/common/config.hpp>
// #include <um2/common/bit_ceil.hpp>

// #include <cstring>
// #include <ostream>
// #include <string>

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

  //    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr char & operator [] (length_t const);
  //
  //    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr char const & operator [] (length_t const) const;
  //
  //    UM2_PURE UM2_HOSTDEV constexpr char * begin() const;
  //
  //    UM2_PURE UM2_HOSTDEV constexpr char * end() const;
  //
  //    UM2_PURE UM2_HOSTDEV constexpr char const * cbegin() const;
  //
  //    UM2_PURE UM2_HOSTDEV constexpr char const * cend() const;
  //
  //    UM2_PURE UM2_HOSTDEV constexpr length_t size() const;
  //
  //    UM2_PURE UM2_HOSTDEV constexpr length_t capacity() const;
  //
  //    UM2_PURE UM2_HOSTDEV constexpr char * data();
  //
  //    UM2_PURE UM2_HOSTDEV constexpr char const * data() const;
  //
  //    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr char & front();
  //
  //    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr char const & front() const;
  //
  //    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr char & back();
  //
  //    UM2_NDEBUG_PURE UM2_HOSTDEV constexpr char const & back() const;
  //
  //    // -- Constructors --
  //
  //    UM2_HOSTDEV constexpr String() = default;
  //
  //    UM2_HOSTDEV String(length_t const);
  //
  //    UM2_HOSTDEV String(length_t const, char const);
  //
  //    UM2_HOSTDEV String(String const &);
  //
  //    template <size_t N>
  //    UM2_HOSTDEV String(char const (&)[N]);
  //
  //    String(std::string const &);
  //
  //    // -- Methods --
  //
  //    UM2_HOSTDEV void clear();
  //
  //    UM2_HOSTDEV inline void reserve(length_t);
  //
  //    UM2_HOSTDEV void resize(length_t);
  //
  //    UM2_HOSTDEV void push_back(char const);
  //
  //    UM2_PURE UM2_HOSTDEV constexpr bool empty() const;
  //
  //    UM2_HOSTDEV void insert(char const *, length_t const, char const);
  //
  //    UM2_HOSTDEV void insert(char const *, char const);
  //
  //    UM2_PURE UM2_HOSTDEV constexpr bool contains(char const) const;
  //
  //    UM2_PURE constexpr bool starts_with(std::string const &) const;
  //
  //    UM2_PURE constexpr bool ends_with(std::string const &) const;
  //
  //    // -- Operators --
  //
  //    UM2_HOSTDEV String & operator = (String const &);
  //
  //    UM2_PURE UM2_HOSTDEV constexpr bool operator == (String const &) const;
  //
  //    template <size_t N>
  //    UM2_HOSTDEV String & operator = (char const (&)[N]);
  //
  //    String & operator = (std::string const &);
  //
  //    UM2_PURE UM2_HOSTDEV constexpr bool operator < (String const &) const;
  //
  //    template <size_t N>
  //    UM2_PURE UM2_HOSTDEV constexpr bool operator == (char const (&)[N]) const;
  //
  //    UM2_PURE constexpr bool operator == (std::string const &) const;

}; // struct String

// -- Methods --

// UM2_PURE std::string to_string(String const & v);
//
//// -- IO --
//
// std::ostream & operator << (std::ostream &, String const &);

} // namespace um2

#include "string.inl"
