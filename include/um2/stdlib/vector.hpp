#pragma once

#include <um2/stdlib/algorithm.hpp> // copy
#include <um2/stdlib/math.hpp>      // max
#include <um2/stdlib/memory.hpp>    // addressof
#include <um2/stdlib/utility.hpp>   // move

#include <initializer_list> // std::initializer_list

//==============================================================================
// VECTOR
//==============================================================================
// An std::vector-like class without and Allocator template parameter.

namespace um2
{

template <typename T>
class Vector
{

  using Ptr = T *;
  using ConstPtr = T const *;

  Ptr _begin = nullptr;
  Ptr _end = nullptr;
  Ptr _end_cap = nullptr;

  //==============================================================================
  // Private methods
  //==============================================================================
  // NOLINTBEGIN(readability-identifier-naming) match std::vector

  // Allocate memory for n elements
  HOSTDEV constexpr void
  allocate(Int n) noexcept;

  // Construct n default-initialized elements at the end of the vector
  HOSTDEV constexpr void
  construct_at_end(Int n) noexcept;

  // Construct n elements at the end of the vector, each with value
  HOSTDEV constexpr void
  construct_at_end(Int n, T const & value) noexcept;

  // Destroy elements at the end of the vector until new_last
  // Does not change capacity
  // _begin <= new_last <= _end
  HOSTDEV constexpr void
  destruct_at_end(Ptr new_last) noexcept;

  // Grow the capacity of the vector by n elements
  // Retains the values of the elements already in the vector
  HOSTDEV constexpr void
  grow(Int n) noexcept;

  // Return the recommended capacity for a vector of size new_size.
  // Either double the current capacity or use the new_size if it is larger.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  recommend(Int new_size) const noexcept -> Int;

  // Append n default-initialized elements to the end of the vector
  // Grows the capacity of the vector if necessary
  // Retains the values of the elements already in the vector
  HOSTDEV constexpr void
  append_default(Int n) noexcept;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Vector() noexcept = default;

  HOSTDEV explicit constexpr Vector(Int n) noexcept;

  HOSTDEV constexpr Vector(Int n, T const & value) noexcept;

  HOSTDEV constexpr Vector(Vector const & v) noexcept;

  HOSTDEV constexpr Vector(Vector && v) noexcept;

  HOSTDEV constexpr Vector(T const * first, T const * last) noexcept;

  HOSTDEV constexpr Vector(std::initializer_list<T> const & list) noexcept;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV constexpr ~Vector() noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  max_size() noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Int;

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
  resize(Int n) noexcept;

  HOSTDEV constexpr void
  reserve(Int n) noexcept;

  HOSTDEV constexpr void
  push_back(T const & value) noexcept;

  HOSTDEV constexpr void
  push_back(T && value) noexcept;

  HOSTDEV constexpr void
  push_back(Int n, T const & value) noexcept;

  template <typename... Args>
  HOSTDEV constexpr void
  emplace_back(Args &&... args) noexcept;

  HOSTDEV constexpr void
  pop_back() noexcept;

  //==============================================================================
  // Operators
  //==============================================================================

  PURE HOSTDEV constexpr auto
  operator[](Int i) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator[](Int i) const noexcept -> T const &;

  HOSTDEV constexpr auto
  operator=(Vector const & v) noexcept -> Vector &;

  HOSTDEV constexpr auto
  operator=(Vector && v) noexcept -> Vector &;

  HOSTDEV constexpr auto
  operator=(std::initializer_list<T> const & list) noexcept -> Vector &;

  PURE constexpr auto
  operator==(Vector const & v) const noexcept -> bool;

  // NOLINTEND(readability-identifier-naming)
}; // class Vector

// Vector<bool> is a specialization that is not supported
template <>
class Vector<bool>
{
};

//==============================================================================
// Private methods
//==============================================================================

// Allocate memory for n elements
template <class T>
HOSTDEV constexpr void
Vector<T>::allocate(Int n) noexcept
{
  ASSERT(n < max_size());
  ASSERT(_begin == nullptr);
  _begin = static_cast<T *>(::operator new(static_cast<size_t>(n) * sizeof(T)));
  _end = _begin;
  _end_cap = _begin + n;
}

// Construct n default-initialized elements at the end of the vector
template <class T>
HOSTDEV constexpr void
Vector<T>::construct_at_end(Int n) noexcept
{
  Ptr new_end = _end + n;
  for (Ptr pos = _end; pos != new_end; ++pos) {
    um2::construct_at(pos);
  }
  _end = new_end;
}

// Construct n elements with value at the end of the vector
template <class T>
HOSTDEV constexpr void
Vector<T>::construct_at_end(Int n, T const & value) noexcept
{
  Ptr new_end = _end + n;
  for (Ptr pos = _end; pos != new_end; ++pos) {
    um2::construct_at(pos, value);
  }
  _end = new_end;
}

// Destroy elements at the end of the vector until new_last
template <class T>
HOSTDEV constexpr void
Vector<T>::destruct_at_end(Ptr new_last) noexcept
{
  Ptr soon_to_be_end = _end;
  while (new_last != soon_to_be_end) {
    um2::destroy_at(--soon_to_be_end);
  }
  _end = new_last;
}

// Return the recommended capacity for a new size. Either double the current
// capacity or use the new size if it is larger.
template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::recommend(Int new_size) const noexcept -> Int
{
  return um2::max(2 * capacity(), new_size);
}

// Grow the vector by n elements
template <class T>
HOSTDEV constexpr void
Vector<T>::grow(Int n) noexcept
{
  ASSERT_ASSUME(n > 0);
  Int const current_size = size();
  Int const new_size = current_size + n;
  Int const new_capacity = recommend(new_size);
  Ptr new_begin =
      static_cast<T *>(::operator new(static_cast<size_t>(new_capacity) * sizeof(T)));
  Ptr new_end = new_begin;
  // Move each old element to the new buffer
  for (Ptr old_pos = _begin; old_pos != _end; ++old_pos, ++new_end) {
    um2::construct_at(new_end, um2::move(*old_pos));
  }
  // Destroy the old elements
  destruct_at_end(_begin);
  // Update the pointers
  delete _begin;
  _begin = new_begin;
  _end = new_end;
  _end_cap = _begin + new_capacity;
}

// Append n default-initialized elements to the end of the vector
template <class T>
HOSTDEV constexpr void
Vector<T>::append_default(Int n) noexcept
{
  // If we have enough capacity, just construct the new elements
  // Otherwise, allocate a new buffer and move the elements over
  if (static_cast<Int>(_end_cap - _end) < n) {
    grow(n);
  }
  // Construct the new elements
  construct_at_end(n);
}

//==============================================================================
// Constructors
//==============================================================================

// Default construct n elements
template <class T>
HOSTDEV constexpr Vector<T>::Vector(Int const n) noexcept
{
  allocate(n);
  construct_at_end(n);
}

// Construct n elements with value
template <class T>
HOSTDEV constexpr Vector<T>::Vector(Int const n, T const & value) noexcept
{
  allocate(n);
  construct_at_end(n, value);
}

// Copy construct from a vector
template <class T>
HOSTDEV constexpr Vector<T>::Vector(Vector<T> const & v) noexcept
{
  Int const n = v.size();
  allocate(n);
  construct_at_end(n);
  copy(v._begin, v._end, _begin);
}

// Move construct from a vector
template <class T>
HOSTDEV constexpr Vector<T>::Vector(Vector<T> && v) noexcept
    : _begin{v._begin},
      _end{v._end},
      _end_cap{v._end_cap}
{
  v._begin = nullptr;
  v._end = nullptr;
  v._end_cap = nullptr;
}

template <class T>
HOSTDEV constexpr Vector<T>::Vector(T const * first, T const * last) noexcept
{
  Int const n = static_cast<Int>(last - first);
  allocate(n);
  construct_at_end(n);
  copy(first, last, _begin);
}

// Construct from an initializer list
template <class T>
HOSTDEV constexpr Vector<T>::Vector(std::initializer_list<T> const & list) noexcept
{
  // Initializer lists can't be moved from, so we have to copy.
  Int const n = static_cast<Int>(list.size());
  allocate(n);
  construct_at_end(n);
  copy(list.begin(), list.end(), _begin);
}

//==============================================================================-
// Destructor
//==============================================================================-

template <class T>
HOSTDEV constexpr Vector<T>::~Vector() noexcept
{
  // If the vector is not empty, destroy the elements and deallocate the buffer
  if (_begin != nullptr) {
    this->destruct_at_end(_begin);
    ::operator delete(_begin);
  }
}

//==============================================================================-
// Accessors
//==============================================================================-

// Return the maximum number of elements the vector can hold
template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::max_size() noexcept -> Int
{
  return intMax() / sizeof(T);
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::begin() noexcept -> T *
{
  return _begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::begin() const noexcept -> T const *
{
  return _begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::end() noexcept -> T *
{
  return _end;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::end() const noexcept -> T const *
{
  return _end;
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::size() const noexcept -> Int
{
  return static_cast<Int>(_end - _begin);
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::capacity() const noexcept -> Int
{
  return static_cast<Int>(_end_cap - _begin);
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::empty() const noexcept -> bool
{
  return _begin == _end;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::cbegin() const noexcept -> T const *
{
  return begin();
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::cend() const noexcept -> T const *
{
  return end();
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::front() noexcept -> T &
{
  return *_begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::front() const noexcept -> T const &
{
  return *_begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::back() noexcept -> T &
{
  ASSERT(size() > 0);
  return *(_end - 1);
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::back() const noexcept -> T const &
{
  ASSERT(size() > 0);
  return *(_end - 1);
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::data() noexcept -> T *
{
  return _begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::data() const noexcept -> T const *
{
  return _begin;
}

//==============================================================================-
// Operators
//==============================================================================-

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::operator[](Int const i) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < size());
  return _begin[i];
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::operator[](Int const i) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < size());
  return _begin[i];
}

template <class T>
HOSTDEV constexpr auto
Vector<T>::operator=(Vector<T> const & v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    // If the vector is not empty, destroy the elements and deallocate the buffer
    destruct_at_end(_begin);
    ::operator delete(_begin);
    // Allocate a new buffer and copy the elements
    _begin = nullptr;
    allocate(v.size());
    construct_at_end(v.size());
    copy(v.begin(), v.end(), _begin);
  }
  return *this;
}

template <class T>
HOSTDEV constexpr auto
Vector<T>::operator=(Vector<T> && v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    // If the vector is not empty, destroy the elements and deallocate the buffer
    destruct_at_end(_begin);
    ::operator delete(_begin);
    // Move the buffer from v
    _begin = v._begin;
    _end = v._end;
    _end_cap = v._end_cap;
    v._begin = nullptr;
    v._end = nullptr;
    v._end_cap = nullptr;
  }
  return *this;
}

template <class T>
HOSTDEV constexpr auto
Vector<T>::operator=(std::initializer_list<T> const & list) noexcept -> Vector &
{
  // If the vector is not empty, destroy the elements and deallocate the buffer
  destruct_at_end(_begin);
  ::operator delete(_begin);
  // Allocate a new buffer and copy the elements
  _begin = nullptr;
  allocate(static_cast<Int>(list.size()));
  construct_at_end(static_cast<Int>(list.size()));
  copy(list.begin(), list.end(), _begin);
  return *this;
}

template <class T>
PURE constexpr auto
Vector<T>::operator==(Vector<T> const & v) const noexcept -> bool
{
  return size() == v.size() && std::equal(begin(), end(), v.begin());
}

//==============================================================================
// Methods
//==============================================================================

template <class T>
HOSTDEV constexpr void
Vector<T>::clear() noexcept
{
  um2::destroy(_begin, _end);
  _end = _begin;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::resize(Int const n) noexcept
{
  Int const cs = size();
  // If we are shrinking, destroy the elements that are no longer needed
  // If we are growing, default construct the new elements
  if (cs < n) {
    append_default(n - cs);
  } else if (cs > n) {
    destruct_at_end(_begin + n);
  }
}

template <class T>
HOSTDEV constexpr void
Vector<T>::reserve(Int const n) noexcept
{
  // If we have enough capacity, do nothing.
  // Otherwise, allocate a new buffer and move the elements over
  if (n > capacity()) {
    grow(n - size());
  }
}

template <class T>
HOSTDEV constexpr void
Vector<T>::push_back(T const & value) noexcept
{
  if (_end == _end_cap) {
    grow(1);
  }
  construct_at(_end, value);
  ++_end;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::push_back(T && value) noexcept
{
  if (_end == _end_cap) {
    grow(1);
  }
  um2::construct_at(_end, um2::move(value));
  ++_end;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::push_back(Int const n, T const & value) noexcept
{
  // If we have enough capacity, just construct the new elements
  // Otherwise, allocate a new buffer and move the elements over
  if (static_cast<Int>(_end_cap - _end) < n) {
    grow(n);
  }
  // Construct the new elements
  construct_at_end(n, value);
}

template <class T>
template <class... Args>
HOSTDEV constexpr void
Vector<T>::emplace_back(Args &&... args) noexcept
{
  if (_end == _end_cap) {
    grow(1);
  }
  um2::construct_at(_end, um2::forward<Args>(args)...);
  ++_end;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::pop_back() noexcept
{
  ASSERT(size() > 0);
  um2::destroy_at(--_end);
}

} // namespace um2
