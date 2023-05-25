#pragma once

#include <um2/common/bit_ceil.hpp>
#include <um2/common/config.hpp>

#include <thrust/execution_policy.h>      // thrust::seq
#include <thrust/iterator/zip_iterator.h> // thrust::zip_iterator
#include <thrust/logical.h>               // thrust::all_of
#include <thrust/tuple.h>                 // thrust::tuple

#include <cmath>            // std::abs
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

  UM2_HOSTDEV ~Vector() { delete[] _data; }

  // -- Accessors --

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto size() const -> len_t;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto capacity() const -> len_t;

  UM2_PURE UM2_HOSTDEV constexpr auto data() -> T *;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto data() const -> T const *;

  UM2_PURE UM2_HOSTDEV constexpr auto begin() -> T *;

  UM2_PURE UM2_HOSTDEV constexpr auto end() -> T *;

  UM2_PURE UM2_HOSTDEV constexpr auto cbegin() const -> T const *;

  UM2_PURE UM2_HOSTDEV constexpr auto cend() const -> T const *;

  UM2_PURE UM2_HOSTDEV constexpr auto front() -> T &;

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto front() const -> T const &;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto back() -> T &;

  UM2_NDEBUG_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto back() const -> T const &;

  // -- Constructors --

  constexpr Vector() = default;

  UM2_HOSTDEV explicit Vector(len_t n);

  UM2_HOSTDEV Vector(len_t n, T const & value);

  UM2_HOSTDEV Vector(Vector const & v);

  UM2_HOSTDEV Vector(Vector && v) noexcept;

  // cppcheck-suppress noExplicitConstructor
  Vector(std::initializer_list<T> const & list);

  // -- Methods --

  UM2_HOSTDEV void clear();

  UM2_HOSTDEV inline void reserve(len_t n);

  UM2_HOSTDEV void resize(len_t n);

  UM2_HOSTDEV inline void push_back(T const & value);

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto empty() const -> bool;

  UM2_HOSTDEV void insert(T const * pos, len_t n, T const & value);

  UM2_HOSTDEV void insert(T const * pos, T const & value);

  UM2_PURE UM2_HOSTDEV [[nodiscard]] constexpr auto contains(T const & value) const
      -> bool;

  // -- Operators --

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) -> T &;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto operator[](len_t i) const -> T const &;

  UM2_HOSTDEV auto operator=(Vector const & v) -> Vector &;

  UM2_HOSTDEV auto operator=(Vector && v) noexcept -> Vector &;

  UM2_PURE UM2_HOSTDEV constexpr auto operator==(Vector const & v) const -> bool;

}; // struct Vector

// -- Methods --

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto is_approx(Vector<T> const & a, Vector<T> const & b,
                                              T const & epsilon = T{}) -> bool;

} // namespace um2

#include "vector.inl"