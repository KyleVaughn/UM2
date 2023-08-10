#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/algorithm.hpp> // copy
#include <um2/stdlib/math.hpp>      // min
#include <um2/stdlib/memory.hpp>    // addressof
#include <um2/stdlib/utility.hpp>   // move

#include <cstring> // memcpy
#include <string>

namespace um2
{

// -----------------------------------------------------------------------------
// SHORT STRING
// -----------------------------------------------------------------------------
// A stack allocated string that can hold up to 31 characters.

struct ShortString {

private:
  char _c[32];
  // data[31] is used to store the remaining capacity of the string.
  // In the event that the array is full, data[31] will be 0, a null terminator.

public:
  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr ShortString() noexcept;

  template <uint64_t N>
  HOSTDEV constexpr explicit ShortString(char const (&s)[N]) noexcept;

  HOSTDEV constexpr explicit ShortString(char const * s) noexcept;

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  capacity() noexcept -> Size;

  // cppcheck-suppress functionConst
  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> char *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> char const *;

  // -----------------------------------------------------------------------------
  // Operators
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr auto
  operator==(ShortString const & s) const noexcept -> bool;

  HOSTDEV constexpr auto
  operator!=(ShortString const & s) const noexcept -> bool;

  HOSTDEV constexpr auto
  operator<(ShortString const & s) const noexcept -> bool;

  HOSTDEV constexpr auto
  operator<=(ShortString const & s) const noexcept -> bool;

  HOSTDEV constexpr auto
  operator>(ShortString const & s) const noexcept -> bool;

  HOSTDEV constexpr auto
  operator>=(ShortString const & s) const noexcept -> bool;

  constexpr auto
  operator==(std::string const & s) const noexcept -> bool;

  HOSTDEV constexpr auto
  // cppcheck-suppress functionConst
  operator[](Size i) noexcept -> char &;

  HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> char const &;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  HOSTDEV [[nodiscard]] constexpr auto
  compare(ShortString const & s) const noexcept -> int;

}; // struct ShortString

} // namespace um2

#include "ShortString.inl"
