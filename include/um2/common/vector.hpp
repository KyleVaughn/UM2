#pragma once

#include <um2/memory.hpp> // Allocator

#include <cuda/std/bit> // cuda::std::bit_ceil

//#include <cmath>            // std::abs
//#include <cstring>          // memcpy
//#include <initializer_list> // std::initializer_list

namespace um2
{

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------
// An std::vector-like class. Allocates 2^N elements, where N is the smallest 
// integer such that 2^N >= size.
//
// https://en.cppreference.com/w/cpp/container/vector

template <typename T, typename Allocator = Allocator<T>>
struct Vector {

  using Ptr = T *;

  private:
    Ptr _begin = nullptr;
    Ptr _end = nullptr;
    Ptr _end_cap = nullptr;

public:
  // -----------------------------------------------------------------------------
  // Destructor
  // -----------------------------------------------------------------------------

  HOSTDEV ~Vector() noexcept { destroy(_begin, _end); } 
//
//  // -----------------------------------------------------------------------------
//  // Accessors
//  // -----------------------------------------------------------------------------
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto size() const noexcept -> len_t;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto capacity() const noexcept -> len_t;
//
//  // cppcheck-suppress functionConst
//  PURE HOSTDEV constexpr auto data() noexcept -> T *;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto data() const noexcept -> T const *;
//
//  // cppcheck-suppress functionConst
//  PURE HOSTDEV constexpr auto begin() noexcept -> T *;
//
//  // cppcheck-suppress functionConst
//  PURE HOSTDEV constexpr auto end() noexcept -> T *;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto cbegin() const noexcept -> T const *;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto cend() const noexcept -> T const *;
//
//  // cppcheck-suppress functionConst
//  PURE HOSTDEV constexpr auto front() noexcept -> T &;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto front() const noexcept -> T const &;
//
//  NDEBUG_PURE HOSTDEV constexpr auto back() noexcept -> T &;
//
//  NDEBUG_PURE HOSTDEV [[nodiscard]] constexpr auto back() const noexcept
//      -> T const &;
//
//  // -----------------------------------------------------------------------------
//  // Constructors
//  // -----------------------------------------------------------------------------
//
//  constexpr Vector() = default;
//
//  HOSTDEV explicit Vector(len_t n);
//
//  HOSTDEV Vector(len_t n, T const & value);
//
//  HOSTDEV Vector(Vector const & v);
//
//  HOSTDEV Vector(Vector && v) noexcept;
//
//  // cppcheck-suppress noExplicitConstructor
//  // NOLINTNEXTLINE(google-explicit-constructor)
//  Vector(std::initializer_list<T> const & list);
//
//  // -----------------------------------------------------------------------------
//  // Methods
//  // -----------------------------------------------------------------------------
//
//  HOSTDEV void clear() noexcept;
//
//  HOSTDEV inline void reserve(len_t n);
//
//  HOSTDEV void resize(len_t n);
//
//  HOSTDEV inline void push_back(T const & value);
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto empty() const -> bool;
//
//  HOSTDEV void insert(T const * pos, len_t n, T const & value);
//
//  HOSTDEV void insert(T const * pos, T const & value);
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  contains(T const & value) const noexcept -> bool requires(!std::floating_point<T>);
//
//  // -----------------------------------------------------------------------------
//  // Operators
//  // -----------------------------------------------------------------------------
//
//  NDEBUG_PURE HOSTDEV constexpr auto operator[](len_t i) noexcept -> T &;
//
//  NDEBUG_PURE HOSTDEV constexpr auto operator[](len_t i) const noexcept
//      -> T const &;
//
//  HOSTDEV auto operator=(Vector const & v) -> Vector &;
//
//  HOSTDEV auto operator=(Vector && v) noexcept -> Vector &;
//
//  PURE HOSTDEV constexpr auto operator==(Vector const & v) const noexcept -> bool;

}; // struct Vector

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

//template <typename T>
//requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) PURE HOSTDEV
//    constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                            T epsilon = T{}) noexcept -> bool;
//
//template <typename T>
//requires(std::unsigned_integral<T>) PURE HOSTDEV
//    constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                            T epsilon = T{}) noexcept -> bool;

} // namespace um2

#include "vector.inl"
