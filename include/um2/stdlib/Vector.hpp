#pragma once

#include <um2/stdlib/algorithm.hpp> // copy
#include <um2/stdlib/math.hpp>      // max
#include <um2/stdlib/memory.hpp>    // addressof
#include <um2/stdlib/utility.hpp>   // move

#include <initializer_list> // std::initializer_list

namespace um2
{

//==============================================================================
// VECTOR
//==============================================================================
// An std::vector-like class without and Allocator template parameter.
//
// https://en.cppreference.com/w/cpp/container/vector

template <typename T>
struct Vector {

  using Ptr = T *;
  using ConstPtr = T const *;

private:
  Ptr _begin = nullptr;
  Ptr _end = nullptr;
  Ptr _end_cap = nullptr;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Vector() noexcept = default;

  HOSTDEV explicit constexpr Vector(Size n) noexcept;

  HOSTDEV constexpr Vector(Size n, T const & value) noexcept;

  HOSTDEV constexpr Vector(Vector const & v) noexcept;

  HOSTDEV constexpr Vector(Vector && v) noexcept;

  HOSTDEV constexpr Vector(std::initializer_list<T> const & list) noexcept;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV constexpr ~Vector() noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  // NOLINTBEGIN(readability-identifier-naming) justification: match stdlib

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  max_size() noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  empty() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cbegin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cend() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  front() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  front() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  back() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  back() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> T const *;

  //==============================================================================
  // Methods
  //==============================================================================

  HOSTDEV constexpr void
  clear() noexcept;

  HOSTDEV constexpr void
  resize(Size n) noexcept;

  HOSTDEV constexpr void
  push_back(T const & value) noexcept;

  HOSTDEV constexpr void
  push_back(T && value) noexcept;

  HOSTDEV constexpr void
  push_back(Size n, T const & value) noexcept;

  //==============================================================================
  // Operators
  //==============================================================================

  PURE HOSTDEV constexpr auto
  operator[](Size i) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> T const &;

  HOSTDEV constexpr auto
  operator=(Vector const & v) noexcept -> Vector &;

  HOSTDEV constexpr auto
  operator=(Vector && v) noexcept -> Vector &;

  HOSTDEV constexpr auto
  operator=(std::initializer_list<T> const & list) noexcept -> Vector &;

  constexpr auto
  operator==(Vector const & v) const noexcept -> bool;

  //==============================================================================
  // Hidden
  //==============================================================================

  HOSTDEV HIDDEN constexpr void
  allocate(Size n) noexcept;

  HOSTDEV HIDDEN constexpr void
  construct_at_end(Size n) noexcept;

  HOSTDEV HIDDEN constexpr void
  construct_at_end(Size n, T const & value) noexcept;

  HOSTDEV HIDDEN constexpr void
  destruct_at_end(Ptr new_last) noexcept;

  HOSTDEV HIDDEN constexpr void
  grow(Size n) noexcept;

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  recommend(Size new_size) const noexcept -> Size;

  HOSTDEV HIDDEN constexpr void
  append_default(Size n) noexcept;

  // NOLINTEND(readability-identifier-naming)
}; // struct Vector

// Vector<bool> is a specialization that is not supported
template <>
struct Vector<bool> {
};

} // namespace um2

#include "Vector.inl"
