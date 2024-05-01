#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/algorithm/min.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/cstring/memcmp.hpp>
#include <um2/stdlib/cstring/strlen.hpp>

//==============================================================================
// STRING VIEW
//==============================================================================

namespace um2
{

//==============================================================================
// Helper functions
//==============================================================================

class StringView
{
public:
  using Ptr = char *;
  using ConstPtr = char const *;

  static constexpr uint64_t npos = static_cast<uint64_t>(-1);

private:
  ConstPtr _data;
  uint64_t _size;

  //==============================================================================
  // Private member functions
  //==============================================================================

public:
  //==============================================================================
  // Constructors and assignment
  //==============================================================================

  HOSTDEV constexpr StringView() noexcept;

  HOSTDEV constexpr StringView(StringView const & s) noexcept = default;

  HOSTDEV constexpr StringView(StringView && s) noexcept = default;

  HOSTDEV constexpr StringView(char const * s, uint64_t size) noexcept;

  // NOLINTNEXTLINE(google-explicit-constructor) match std::string
  HOSTDEV constexpr StringView(char const * s) noexcept;

  HOSTDEV constexpr StringView(char const * begin, char const * end) noexcept;

  HOSTDEV StringView(std::nullptr_t) = delete;

  HOSTDEV constexpr auto
  operator=(StringView const & s) noexcept -> StringView & = default;

  HOSTDEV constexpr auto
  operator=(StringView && s) noexcept -> StringView & = default;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV constexpr ~StringView() = default;

  //==============================================================================
  // Iterators
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> ConstPtr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> ConstPtr;

  //==============================================================================
  // Element access
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  operator[](Int i) const noexcept -> char;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  front() const noexcept -> char const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  back() const noexcept -> char const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> ConstPtr;

  //==============================================================================
  // Capacity
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> uint64_t;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  empty() const noexcept -> bool;

  //==============================================================================
  // Modifiers
  //==============================================================================

  HOSTDEV constexpr void
  // NOLINTBEGIN(readability-identifier-naming) match std::string
  remove_prefix(uint64_t n) noexcept;

  //==============================================================================
  // Operations
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  compare(StringView sv) const noexcept -> int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  compare(uint64_t pos, uint64_t count, StringView sv) const noexcept -> int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  ends_with(StringView sv) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  ends_with(char const * s) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  starts_with(StringView sv) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  starts_with(char const * s) const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  substr(uint64_t pos, uint64_t count) const noexcept -> StringView;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  find_first_of(StringView sv, uint64_t pos = 0) const noexcept -> uint64_t;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  find_first_of(char c, uint64_t pos = 0) const noexcept -> uint64_t;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  find_first_of(char const * s, uint64_t pos, uint64_t count) const noexcept -> uint64_t;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  find_first_of(char const * s, uint64_t pos = 0) const noexcept -> uint64_t;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  find_first_not_of(char c, uint64_t pos = 0) const noexcept -> uint64_t;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  find_last_of(char c, uint64_t pos = npos) const noexcept -> uint64_t;

  // NOLINTEND(readability-identifier-naming) match std::string

  //==============================================================================
  // Non-standard modifiers
  //==============================================================================

  HOSTDEV constexpr void
  removeLeadingSpaces() noexcept;

  HOSTDEV constexpr auto
  getTokenAndShrink(char delim = ' ') noexcept -> StringView;

}; // class StringView

//==============================================================================
// Free functions
//==============================================================================

PURE HOSTDEV constexpr auto
operator==(StringView lhs, StringView rhs) noexcept -> bool
{
  if (lhs.size() != rhs.size()) {
    return false;
  }
  return lhs.compare(rhs) == 0;
}

PURE HOSTDEV constexpr auto
operator!=(StringView lhs, StringView rhs) noexcept -> bool
{
  if (lhs.size() != rhs.size()) {
    return true;
  }
  return lhs.compare(rhs) != 0;
}

PURE HOSTDEV constexpr auto
operator<(StringView lhs, StringView rhs) noexcept -> bool
{
  return lhs.compare(rhs) < 0;
}

PURE HOSTDEV constexpr auto
operator<=(StringView lhs, StringView rhs) noexcept -> bool
{
  return lhs.compare(rhs) <= 0;
}

PURE HOSTDEV constexpr auto
operator>(StringView lhs, StringView rhs) noexcept -> bool
{
  return lhs.compare(rhs) > 0;
}

PURE HOSTDEV constexpr auto
operator>=(StringView lhs, StringView rhs) noexcept -> bool
{
  return lhs.compare(rhs) >= 0;
}

//==============================================================================
// Constructors and assignment
//==============================================================================

HOSTDEV constexpr StringView::StringView() noexcept
    : _data(nullptr),
      _size(0)
{
}

HOSTDEV constexpr StringView::StringView(char const * s, uint64_t size) noexcept
    : _data(s),
      _size(size)
{
  ASSERT(s != nullptr);
  ASSERT(size > 0);
}

HOSTDEV constexpr StringView::StringView(char const * s) noexcept
    : _data(s),
      _size(strlen(s))
{
  ASSERT(s != nullptr);
  ASSERT(_size > 0);
}

HOSTDEV constexpr StringView::StringView(char const * begin, char const * end) noexcept
    : _data(begin),
      _size(static_cast<uint64_t>(end - begin))
{
  ASSERT(begin != nullptr);
  ASSERT(end != nullptr);
  ASSERT(_size > 0);
}

//==============================================================================
// Iterators
//==============================================================================

PURE HOSTDEV constexpr auto
StringView::begin() const noexcept -> ConstPtr
{
  return _data;
}

PURE HOSTDEV constexpr auto
StringView::end() const noexcept -> ConstPtr
{
  return _data + _size;
}

//==============================================================================
// Element access
//==============================================================================

PURE HOSTDEV constexpr auto
StringView::operator[](Int i) const noexcept -> char
{
  ASSERT_ASSUME(i >= 0);
  ASSERT(static_cast<uint64_t>(i) < _size);
  return _data[i];
}

PURE HOSTDEV constexpr auto
StringView::front() const noexcept -> char const &
{
  ASSERT(_size > 0);
  return _data[0];
}

PURE HOSTDEV constexpr auto
StringView::back() const noexcept -> char const &
{
  ASSERT(_size > 0);
  return _data[_size - 1];
}

PURE HOSTDEV constexpr auto
StringView::data() const noexcept -> ConstPtr
{
  return _data;
}

//==============================================================================
// Capacity
//==============================================================================

PURE HOSTDEV constexpr auto
StringView::size() const noexcept -> uint64_t
{
  return _size;
}

PURE HOSTDEV constexpr auto
StringView::empty() const noexcept -> bool
{
  return _size == 0;
}

//==============================================================================
// Modifiers
//==============================================================================

HOSTDEV constexpr void
StringView::remove_prefix(uint64_t n) noexcept
{
  ASSERT(n <= _size);
  _data += n;
  _size -= n;
}

//==============================================================================
// Operations
//==============================================================================

PURE HOSTDEV constexpr auto
StringView::compare(StringView sv) const noexcept -> int
{
  auto const min_size = um2::min(size(), sv.size());
  auto result = um2::memcmp(data(), sv.data(), min_size);
  // If they compare equal, but are different sizes, the longer one is greater
  if (result == 0) {
    result = size() == sv.size() ? 0 : (size() < sv.size() ? -1 : 1);
  }
  return result;
}

PURE HOSTDEV constexpr auto
StringView::compare(uint64_t pos, uint64_t count, StringView sv) const noexcept -> int
{
  return substr(pos, count).compare(sv);
}

PURE HOSTDEV constexpr auto
StringView::ends_with(StringView sv) const noexcept -> bool
{
  return size() >= sv.size() && compare(size() - sv.size(), sv.size(), sv) == 0;
}

PURE HOSTDEV constexpr auto
StringView::ends_with(char const * s) const noexcept -> bool
{
  return ends_with(StringView(s));
}

PURE HOSTDEV constexpr auto
StringView::starts_with(StringView sv) const noexcept -> bool
{
  return size() >= sv.size() && compare(0, sv.size(), sv) == 0;
}

PURE HOSTDEV constexpr auto
StringView::starts_with(char const * s) const noexcept -> bool
{
  return starts_with(StringView(s));
}

PURE HOSTDEV [[nodiscard]] constexpr auto
StringView::substr(uint64_t pos, uint64_t count) const noexcept -> StringView
{
  ASSERT(pos <= size());
  ASSERT(count + pos <= size());
  return {data() + pos, count};
}

PURE HOSTDEV constexpr auto
StringView::find_first_of(char c, uint64_t pos) const noexcept -> uint64_t
{
  for (uint64_t i = pos; i < size(); ++i) {
    if (data()[i] == c) {
      return i;
    }
  }
  return npos;
}

PURE HOSTDEV constexpr auto
StringView::find_first_not_of(char c, uint64_t pos) const noexcept -> uint64_t
{
  for (uint64_t i = pos; i < size(); ++i) {
    if (data()[i] != c) {
      return i;
    }
  }
  return npos;
}

PURE HOSTDEV constexpr auto
StringView::find_last_of(char c, uint64_t pos) const noexcept -> uint64_t
{
  if (pos < size()) {
    ++pos;
  } else {
    pos = size();
  }
  for (ConstPtr p = data() + pos; p != begin();) {
    if (*--p == c) {
      return static_cast<uint64_t>(p - begin());
    }
  }
  return npos;
}

PURE HOSTDEV constexpr auto
StringView::find_first_of(StringView const sv, uint64_t pos) const noexcept -> uint64_t
{
  // Check valid pos
  if (pos > size()) {
    return npos;
  }

  StringView this_substr(data() + pos, size() - pos);

  while (this_substr.size() >= sv.size()) {
    if (um2::memcmp(this_substr.data(), sv.data(), sv.size()) == 0) {
      return pos;
    }
    ++pos;
    this_substr.remove_prefix(1);
  }

  return npos;
}

PURE HOSTDEV constexpr auto
StringView::find_first_of(char const * s, uint64_t pos, uint64_t count) const noexcept
    -> uint64_t
{
  return find_first_of(StringView(s, count), pos);
}

PURE HOSTDEV constexpr auto
StringView::find_first_of(char const * s, uint64_t pos) const noexcept -> uint64_t
{
  return find_first_of(StringView(s), pos);
}

//==============================================================================
// Non-standard modifiers
//==============================================================================

HOSTDEV constexpr void
StringView::removeLeadingSpaces() noexcept
{
  uint64_t const n = find_first_not_of(' ');
  if (n != npos) {
    remove_prefix(n);
  } else {
    _data = nullptr;
    _size = 0;
  }
}

HOSTDEV constexpr auto
StringView::getTokenAndShrink(char delim) noexcept -> StringView
{
  // Find the first non-delimiter character
  uint64_t const n = find_first_not_of(delim);

  // If there are no non-delimiter characters, return an empty string
  // and leave this string unchanged
  if (n == npos) {
    return {};
  }

  // Find the first delimiter character after the non-delimiter character
  uint64_t const m = find_first_of(delim, n);

  // If there are no delimiter characters after the non-delimiter character,
  // return the substring from n to the end of the string and set this string
  // to an empty string
  if (m == npos) {
    auto const result = substr(n, size() - n);
    _data = nullptr;
    _size = 0;
    return result;
  }

  // Otherwise, return the substring from n to m - n and remove the prefix
  // of this string of length m
  auto const result = substr(n, m - n);
  // Omit the delimiter character
  remove_prefix(m + 1);
  return result;
}

} // namespace um2
