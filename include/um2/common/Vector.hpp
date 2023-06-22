#pragma once

#include <um2/common/algorithm.hpp> // copy
#include <um2/common/memory.hpp> // addressof

#include <cuda/std/bit>     // cuda::std::bit_ceil
#include <cuda/std/utility> // cuda::std::pair

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

  HOSTDEV explicit constexpr Vector(Size n);
  
  HOSTDEV constexpr Vector(Size n, T const & value);
  
  HOSTDEV constexpr Vector(Vector const & v);

  HOSTDEV constexpr Vector(Vector && v) noexcept;
  ////
  ////  // cppcheck-suppress noExplicitConstructor
  ////  Vector(std::initializer_list<T> const & list);
  ////

  // -----------------------------------------------------------------------------
  // Destructor
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr ~Vector() noexcept;

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

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

  // HOSTDEV constexpr void
  // reserve(Size n);

  ////  HOSTDEV void clear() noexcept;
  ////
  ////
  ////  HOSTDEV void resize(Size n);
  ////
  ////  HOSTDEV inline void push_back(T const & value);
  ////
  ////  PURE HOSTDEV [[nodiscard]] constexpr auto empty() const -> bool;
  ////
  ////  HOSTDEV void insert(T const * pos, Size n, T const & value);
  ////
  ////  HOSTDEV void insert(T const * pos, T const & value);
  ////
  ////  PURE HOSTDEV [[nodiscard]] constexpr auto
  ////  contains(T const & value) const noexcept -> bool
  /// requires(!std::floating_point<T>);
  ////
  // -----------------------------------------------------------------------------
  // Operators
  // -----------------------------------------------------------------------------

  NDEBUG_PURE HOSTDEV constexpr auto
  // cppcheck-suppress functionConst
  operator[](Size i) noexcept -> T &;

  NDEBUG_PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> T const &;

  HOSTDEV constexpr auto
  operator=(Vector const & v) -> Vector &;

  HOSTDEV constexpr auto
  operator=(Vector && v) noexcept -> Vector &;
  ////
  ////  PURE HOSTDEV constexpr auto operator==(Vector const & v) const noexcept -> bool;
  //

}; // struct Vector

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

// template <typename T>
// requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T epsilon = T{}) noexcept -> bool;
//
// template <typename T>
// requires(std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T epsilon = T{}) noexcept -> bool;

// Vector<bool> is a specialization that is not supported
template <>
struct Vector<bool> {
};

} // namespace um2

#include "Vector.inl"
