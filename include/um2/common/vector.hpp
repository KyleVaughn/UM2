#pragma once

#include <um2/common/bit_ceil.hpp>
#include <um2/common/config.hpp>

#include <thrust/execution_policy.h>      // thrust::seq
#include <thrust/iterator/zip_iterator.h> // thrust::zip_iterator
#include <thrust/logical.h>               // thrust::all_of
#include <thrust/tuple.h>                 // thrust::tuple

#include <cmath>            // std::abs
#include <initializer_list> // std::initializer_list
#include <ostream>          // std::ostream

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

  UM2_PURE UM2_HOSTDEV constexpr len_t size() const;

  UM2_PURE UM2_HOSTDEV constexpr len_t capacity() const;

  UM2_PURE UM2_HOSTDEV constexpr T * data();

  UM2_PURE UM2_HOSTDEV constexpr T const * data() const;

  UM2_PURE UM2_HOSTDEV constexpr T * begin() const;

  UM2_PURE UM2_HOSTDEV constexpr T * end() const;

  UM2_PURE UM2_HOSTDEV constexpr T const * cbegin() const;

  UM2_PURE UM2_HOSTDEV constexpr T const * cend() const;

  UM2_PURE UM2_HOSTDEV constexpr T & front();

  UM2_PURE UM2_HOSTDEV constexpr T const & front() const;

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr T & back();

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr T const & back() const;

  // -- Constructors --

  constexpr Vector() = default;

  UM2_HOSTDEV explicit Vector(len_t);

  UM2_HOSTDEV Vector(len_t, T const &);

  UM2_HOSTDEV Vector(Vector const &);

  UM2_HOSTDEV explicit Vector(std::initializer_list<T> const &);

  // -- Methods --

  UM2_HOSTDEV void clear();

  UM2_HOSTDEV inline void reserve(len_t);

  UM2_HOSTDEV void resize(len_t);

  UM2_HOSTDEV inline void push_back(T const &);

  UM2_PURE UM2_HOSTDEV constexpr bool empty() const;

  UM2_HOSTDEV void insert(T const *, len_t, T const &);

  UM2_HOSTDEV void insert(T const *, T const &);

  UM2_PURE UM2_HOSTDEV constexpr bool contains(T const &) const;

  // -- Operators --

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr T & operator[](len_t);

  UM2_NDEBUG_PURE UM2_HOSTDEV constexpr T const & operator[](len_t) const;

  UM2_HOSTDEV Vector & operator=(Vector const &);

  UM2_PURE UM2_HOSTDEV constexpr bool operator==(Vector const &) const;

}; // struct Vector

// -- IO --

template <typename T>
std::ostream & operator<<(std::ostream &, Vector<T> const &);

// -- Methods --

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr bool is_approx(Vector<T> const & a, Vector<T> const & b,
                                              T const & epsilon = T{});

} // namespace um2

#include "vector.inl"