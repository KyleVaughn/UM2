#pragma once

#include <um2/common/bit_ceil.hpp>
#include <um2/common/config.hpp>

#include <cstring> // memcpy, strcmp
#include <string>  // std::string

namespace um2
{

// -----------------------------------------------------------------------------
// STRING
// -----------------------------------------------------------------------------
// A std::string-like class, but without an allocator template parameter.
// Allocates 2^N elements, where N is the smallest integer such that 2^N >= size.
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

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto begin() const noexcept -> char8_t *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto end() const noexcept -> char8_t *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto cbegin() const noexcept
      -> char8_t const *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto cend() const noexcept
      -> char8_t const *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto size() const noexcept -> len_t;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto capacity() const noexcept -> len_t;

  // cppcheck-suppress functionConst
  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto data() noexcept -> char8_t *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto data() const noexcept
      -> char8_t const *;

  // -- Constructors --

  constexpr String() = default;

  template <size_t N>
  UM2_HOSTDEV explicit String(char const (&s)[N]);

  UM2_HOSTDEV String(String const & s);

  UM2_HOSTDEV String(String && s) noexcept;

  explicit String(std::string const & s);

  // -- Operators --

  UM2_HOSTDEV auto operator=(String const & s) -> String &;

  UM2_HOSTDEV auto operator=(String && s) noexcept -> String &;

  template <size_t N>
  UM2_HOSTDEV auto operator=(char const (&s)[N]) -> String &;

  auto operator=(std::string const & s) -> String &;

  UM2_PURE UM2_HOSTDEV auto operator==(String const & s) const noexcept -> bool;

  template <size_t N>
  UM2_PURE UM2_HOSTDEV auto operator==(char const (&s)[N]) const noexcept -> bool;

  UM2_PURE auto operator==(std::string const & s) const noexcept -> bool;

  UM2_PURE UM2_HOSTDEV auto operator<(String const & s) const noexcept -> bool;

  UM2_PURE UM2_HOSTDEV auto operator>(String const & s) const noexcept -> bool;

  UM2_PURE UM2_HOSTDEV auto operator<=(String const & s) const noexcept -> bool;

  UM2_PURE UM2_HOSTDEV auto operator>=(String const & s) const noexcept -> bool;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) noexcept -> char8_t &;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) const noexcept
      -> char8_t const &;

  // -- Methods --

  UM2_PURE UM2_HOSTDEV [[nodiscard]] auto compare(char8_t const * s) const noexcept
      -> int;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] auto compare(String const & s) const noexcept -> int;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto contains(char c) const noexcept
      -> bool;

  UM2_PURE [[nodiscard]] constexpr auto starts_with(std::string const & s) const noexcept
      -> bool;

  UM2_PURE [[nodiscard]] constexpr auto ends_with(std::string const & s) const noexcept
      -> bool;

}; // struct String

// -- Methods --

UM2_PURE auto to_string(String const & s) -> std::string;

} // namespace um2

#include "string.inl"
