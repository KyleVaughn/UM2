#pragma once

#include <um2/common/algorithm.hpp> // copy
#include <um2/common/memory.hpp>    // addressof
#include <um2/common/utility.hpp>   // move
#include <um2/math/math_functions.hpp> // um2::max

#include <initializer_list> // std::initializer_list

namespace um2
{

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------
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
  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr Vector() noexcept = default;

  HOSTDEV explicit constexpr Vector(Size n) noexcept;

  HOSTDEV constexpr Vector(Size n, T const & value) noexcept;

  HOSTDEV constexpr Vector(Vector const & v) noexcept;

  HOSTDEV constexpr Vector(Vector && v) noexcept;

  // cppcheck-suppress noExplicitConstructor
  HOSTDEV constexpr Vector(std::initializer_list<T> const & list) noexcept;

  // -----------------------------------------------------------------------------
  // Destructor
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr ~Vector() noexcept;

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  // NOLINTNEXTLINE(readability-identifier-naming)
  max_size() noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
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
  // cppcheck-suppress functionConst
  front() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  front() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  back() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  back() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  data() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> T const *;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr void
  clear() noexcept;

  HOSTDEV constexpr void
  resize(Size n) noexcept;

  // -----------------------------------------------------------------------------
  // Operators
  // -----------------------------------------------------------------------------

  PURE HOSTDEV constexpr auto
  // cppcheck-suppress functionConst
  operator[](Size i) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> T const &;

  HOSTDEV constexpr auto
  operator=(Vector const & v) noexcept -> Vector &;

  HOSTDEV constexpr auto
  operator=(Vector && v) noexcept -> Vector &;

  // -----------------------------------------------------------------------------
  // Hidden
  // -----------------------------------------------------------------------------

  HOSTDEV HIDDEN constexpr void
  allocate(Size n) noexcept;

  HOSTDEV HIDDEN constexpr void
  // NOLINTNEXTLINE(readability-identifier-naming)
  construct_at_end(Size n) noexcept;

  HOSTDEV HIDDEN constexpr void
  // NOLINTNEXTLINE(readability-identifier-naming)
  construct_at_end(Size n, T const & value) noexcept;

  HOSTDEV HIDDEN constexpr void
  // NOLINTNEXTLINE(readability-identifier-naming)
  destruct_at_end(Ptr new_last) noexcept;

  HOSTDEV HIDDEN constexpr void
  // NOLINTNEXTLINE(readability-identifier-naming)
  append_default(Size n) noexcept;

  PURE HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  recommend(Size new_size) const noexcept -> Size;

}; // struct Vector

// Vector<bool> is a specialization that is not supported
template <>
struct Vector<bool> {
};

} // namespace um2

#include "Vector.inl"
