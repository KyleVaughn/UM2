#pragma once

#include <um2/common/bit_ceil.hpp>

#include <cmath>            // std::abs
#include <cstring>          // memcpy
#include <initializer_list> // std::initializer_list

namespace um2
{

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------
// An std::vector-like class, but without an allocator template parameter.
// Allocates 2^N elements, where N is the smallest integer such that 2^N >= size.
// Use this in place of std::vector for hostdev/device code.
//
// https://en.cppreference.com/w/cpp/container/vector

template <typename T>
struct Vector {

private:
  len_t _size = 0;
  len_t _capacity = 0;
  T * _data = nullptr;

public:
  // -- Destructor --

  UM2_HOSTDEV ~Vector() noexcept { delete[] _data; }

  // -- Accessors --

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto size() const noexcept -> len_t;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto capacity() const noexcept -> len_t;

  // cppcheck-suppress functionConst
  UM2_PURE UM2_HOSTDEV constexpr auto data() noexcept -> T *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto data() const noexcept -> T const *;

  // cppcheck-suppress functionConst
  UM2_PURE UM2_HOSTDEV constexpr auto begin() noexcept -> T *;

  // cppcheck-suppress functionConst
  UM2_PURE UM2_HOSTDEV constexpr auto end() noexcept -> T *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto cbegin() const noexcept -> T const *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto cend() const noexcept -> T const *;

  // cppcheck-suppress functionConst
  UM2_PURE UM2_HOSTDEV constexpr auto front() noexcept -> T &;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto front() const noexcept -> T const &;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto back() noexcept -> T &;

  UM2_NDEBUG_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto back() const noexcept
      -> T const &;

  // -- Constructors --

  constexpr Vector() = default;

  UM2_HOSTDEV explicit Vector(len_t n);

  UM2_HOSTDEV Vector(len_t n, T const & value);

  UM2_HOSTDEV Vector(Vector const & v);

  UM2_HOSTDEV Vector(Vector && v) noexcept;

  // cppcheck-suppress noExplicitConstructor
  // NOLINTNEXTLINE(google-explicit-constructor)
  Vector(std::initializer_list<T> const & list);

  // -- Methods --

  UM2_HOSTDEV void clear() noexcept;

  UM2_HOSTDEV inline void reserve(len_t n);

  UM2_HOSTDEV void resize(len_t n);

  UM2_HOSTDEV inline void push_back(T const & value);

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto empty() const -> bool;

  UM2_HOSTDEV void insert(T const * pos, len_t n, T const & value);

  UM2_HOSTDEV void insert(T const * pos, T const & value);

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto
  contains(T const & value) const noexcept -> bool requires(!std::floating_point<T>);

  // -- Operators --

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) noexcept -> T &;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) const noexcept
      -> T const &;

  UM2_HOSTDEV auto operator=(Vector const & v) -> Vector &;

  UM2_HOSTDEV auto operator=(Vector && v) noexcept -> Vector &;

  UM2_PURE UM2_HOSTDEV constexpr auto operator==(Vector const & v) const noexcept -> bool;

}; // struct Vector

// -- Methods --

template <typename T>
requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) UM2_PURE UM2_HOSTDEV
    constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
                            T epsilon = T{}) noexcept -> bool;

template <typename T>
requires(std::unsigned_integral<T>) UM2_PURE UM2_HOSTDEV
    constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
                            T epsilon = T{}) noexcept -> bool;

} // namespace um2

#include "vector.inl"
