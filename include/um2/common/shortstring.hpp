#pragma once

#include <um2/stdlib/algorithm.hpp> // copy
#include <um2/stdlib/math.hpp>      // min
#include <um2/stdlib/memory.hpp>    // addressof
#include <um2/stdlib/utility.hpp>   // move

#include <cstring> // memset
#include <string>

namespace um2
{

//==============================================================================
// SHORT STRING
//==============================================================================
//
// A stack allocated string that can hold up to 31 characters.

class ShortString {

  char _c[32];
  // data[31] is used to store the remaining capacity of the string.
  // In the event that the array is full, data[31] will be 0, a null terminator.

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  HOSTDEV constexpr ShortString() noexcept;

  // We want to allow implicit conversions from string literals.
  // NOLINTBEGIN(google-explicit-constructor) justified above.
  template <uint64_t N>
  HOSTDEV constexpr ShortString(char const (&s)[N]) noexcept;

  HOSTDEV constexpr ShortString(char const * s) noexcept;
  // NOLINTEND(google-explicit-constructor)

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  capacity() noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> char *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> char const *;

  //==============================================================================
  // Operators
  //==============================================================================

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

  template <uint64_t N>
  HOSTDEV constexpr auto
  operator==(char const (&s)[N]) const noexcept -> bool;

  HOSTDEV constexpr auto
  operator[](Size i) noexcept -> char &;

  HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> char const &;

  //==============================================================================
  // Methods
  //==============================================================================

  HOSTDEV [[nodiscard]] constexpr auto
  compare(ShortString const & s) const noexcept -> int;

}; // class ShortString

//==============================================================================
// Constructors
//==============================================================================

HOSTDEV constexpr ShortString::ShortString() noexcept
{
  memset(data(), 0, sizeof(ShortString) - 1);
  _c[31] = static_cast<char>(31);
}

template <uint64_t N>
HOSTDEV constexpr ShortString::ShortString(char const (&s)[N]) noexcept
{
  static_assert(N - 1 <= capacity(), "String too long");
  copy(addressof(s[0]), addressof(s[N]), addressof(_c[0]));
  _c[31] = capacity() - static_cast<char>(N - 1);
  ASSERT(_c[N - 1] == '\0');
}

HOSTDEV constexpr ShortString::ShortString(char const * s) noexcept
{
  Size n = 0;
  while (s[n] != '\0') {
    ++n;
  }
  ASSERT_ASSUME(n <= capacity());
  copy(addressof(s[0]), addressof(s[n + 1]), addressof(_c[0]));
  _c[31] = static_cast<char>(capacity() - n);
  ASSERT(_c[n] == '\0');
}

//==============================================================================
// Accessors
//==============================================================================

PURE HOSTDEV constexpr auto
ShortString::size() const noexcept -> Size
{
  return capacity() - _c[31];
}

PURE HOSTDEV constexpr auto
ShortString::capacity() noexcept -> Size
{
  return sizeof(ShortString) - 1;
}

PURE HOSTDEV constexpr auto
ShortString::data() noexcept -> char *
{
  return addressof(_c[0]);
}

PURE HOSTDEV constexpr auto
ShortString::data() const noexcept -> char const *
{
  return addressof(_c[0]);
}

//==============================================================================
// Operators
//==============================================================================

HOSTDEV constexpr auto
ShortString::operator==(ShortString const & s) const noexcept -> bool
{
  Size const l_size = size();
  Size const r_size = s.size();
  if (l_size != r_size) {
    return false;
  }
  char const * l_data = data();
  char const * r_data = s.data();
  for (Size i = 0; i < l_size; ++i) {
    if (*l_data != *r_data) {
      return false;
    }
    ++l_data;
    ++r_data;
  }
  return true;
}

constexpr auto
ShortString::operator==(std::string const & s) const noexcept -> bool
{
  Size const l_size = size();
  auto const r_size = static_cast<Size>(s.size());
  if (l_size != r_size) {
    return false;
  }
  char const * l_data = data();
  char const * r_data = s.data();
  for (Size i = 0; i < l_size; ++i) {
    if (*l_data != *r_data) {
      return false;
    }
    ++l_data;
    ++r_data;
  }
  return true;
}

template <uint64_t N>
HOSTDEV constexpr auto
ShortString::operator==(char const (&s)[N]) const noexcept -> bool
{
  static_assert(N - 1 <= capacity(), "String too long");
  Size const l_size = size();
  if (l_size != N - 1) {
    return false;
  }
  char const * l_data = data();
  char const * r_data = addressof(s[0]);
  for (Size i = 0; i < l_size; ++i) {
    if (*l_data != *r_data) {
      return false;
    }
    ++l_data;
    ++r_data;
  }
  return true;
}

HOSTDEV constexpr auto
ShortString::operator!=(ShortString const & s) const noexcept -> bool
{
  return !(*this == s);
}

HOSTDEV constexpr auto
ShortString::operator<(ShortString const & s) const noexcept -> bool
{
  return compare(s) < 0;
}

HOSTDEV constexpr auto
ShortString::operator<=(ShortString const & s) const noexcept -> bool
{
  return compare(s) <= 0;
}

HOSTDEV constexpr auto
ShortString::operator>(ShortString const & s) const noexcept -> bool
{
  return compare(s) > 0;
}

HOSTDEV constexpr auto
ShortString::operator>=(ShortString const & s) const noexcept -> bool
{
  return compare(s) >= 0;
}

HOSTDEV constexpr auto
ShortString::operator[](Size const i) noexcept -> char &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < size());
  return _c[i];
}

HOSTDEV constexpr auto
ShortString::operator[](Size const i) const noexcept -> char const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < size());
  return _c[i];
}

//==============================================================================
// Methods
//==============================================================================

HOSTDEV constexpr auto
ShortString::compare(ShortString const & s) const noexcept -> int
{
  Size const l_size = size();
  Size const r_size = s.size();
  Size const min_size = um2::min(l_size, r_size);
  char const * l_data = data();
  char const * r_data = s.data();
  for (Size i = 0; i < min_size; ++i) {
    if (*l_data != *r_data) {
      return static_cast<int>(*l_data) - static_cast<int>(*r_data);
    }
    ++l_data;
    ++r_data;
  }
  return static_cast<int>(l_size) - static_cast<int>(r_size);
}

} // namespace um2
