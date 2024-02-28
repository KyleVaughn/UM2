#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/cstring/strlen.hpp>

//==============================================================================
// STRING VIEW
//==============================================================================

namespace um2
{

class StringView
{
  public:
  using Ptr = char *;
  using ConstPtr = char const *;

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

}; // class StringView

//==============================================================================
// Constructors and assignment
//==============================================================================

HOSTDEV constexpr StringView::StringView() noexcept
    : _data(nullptr), _size(0)
{
}

HOSTDEV constexpr StringView::StringView(char const * s, uint64_t size) noexcept
    : _data(s), _size(size)
{
  ASSERT(s != nullptr);
  ASSERT(size > 0);
}

HOSTDEV constexpr StringView::StringView(char const * s) noexcept
    : _data(s), _size(strlen(s))
{
  ASSERT(s != nullptr);
  ASSERT(_size > 0);
}

HOSTDEV constexpr StringView::StringView(char const * begin, char const * end) noexcept
    : _data(begin), _size(static_cast<uint64_t>(end - begin))
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

} // namespace um2
