namespace um2
{
// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Size const n) noexcept
{
  allocate(n);
  construct_at_end(n);
}

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Size const n, T const & value) noexcept
{
  allocate(n);
  construct_at_end(n, value);
}

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Vector<T> const & v) noexcept
{
  Size const n = v.size();
  allocate(n);
  construct_at_end(n);
  copy(v._begin, v._end, _begin);
}

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
HOSTDEV constexpr Vector<T>::Vector(std::initializer_list<T> const & list) noexcept
{
  Size const n = static_cast<Size>(list.size());
  allocate(n);
  construct_at_end(n);
  copy(list.begin(), list.end(), _begin);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

template <class T>
HOSTDEV constexpr Vector<T>::~Vector() noexcept
{
  if (_begin != nullptr) {
    destruct_at_end(_begin);
    ::operator delete(_begin);
  }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::max_size() noexcept -> Size
{
  return sizeMax() / sizeof(T);
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
Vector<T>::size() const noexcept -> Size
{
  return static_cast<Size>(_end - _begin);
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::capacity() const noexcept -> Size
{
  return static_cast<Size>(_end_cap - _begin);
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
  assert(size() > 0);
  return *(_end - 1);
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::back() const noexcept -> T const &
{
  assert(size() > 0);
  return *(_end - 1);
}

template <class T>
// cppcheck-suppress functionConst
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

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::operator[](Size const i) noexcept -> T &
{
  assert(i < size());
  return _begin[i];
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::operator[](Size const i) const noexcept -> T const &
{
  assert(i < size());
  return _begin[i];
}

template <class T>
HOSTDEV constexpr auto
Vector<T>::operator=(Vector<T> const & v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    destruct_at_end(_begin);
    ::operator delete(_begin);
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
    destruct_at_end(_begin);
    ::operator delete(_begin);
    _begin = v._begin;
    _end = v._end;
    _end_cap = v._end_cap;
    v._begin = nullptr;
    v._end = nullptr;
    v._end_cap = nullptr;
  }
  return *this;
}

// template <class T>
// PURE HOSTDEV constexpr auto
// Vector<T>::operator==(Vector<T> const & v) const noexcept -> bool
//{
//  if (_size != v._size) {
//    return false;
//  }
//  for (Size i = 0; i < _size; ++i) {
//    if (_data[i] != v._data[i]) {
//      return false;
//    }
//  }
//  return true;
//}

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

// template <class T>
// constexpr void
// Vector<T>::reserve(Size n)
//{
//     if (n > capacity()) {
//         if (n > max_size())
//             this->__throw_length_error();
//         allocator_type& __a = this->__alloc();
//         __split_buffer<value_type, allocator_type&> __v(__n, size(), __a);
//         __swap_out_circular_buffer(__v);
//     }
// }

template <class T>
HOSTDEV constexpr void
Vector<T>::clear() noexcept
{
  destroy(_begin, _end);
  _end = _begin;
}
//
// template <class T>
// HOSTDEV inline void Vector<T>::reserve(Size n)
//{
//  if (_capacity < n) {
//    // since n > 0, we are safe to cast to unsigned and use bit_ceil
//    // to determine the next power of 2
//    n = bit_ceil(n);
//    T * new_data = new T[static_cast<Size>(n)];
//    if constexpr (std::is_trivially_copyable_v<T>) {
//      memcpy(new_data, _data, static_cast<Size>(_size) * sizeof(T));
//    } else {
//      for (Size i = 0; i < _size; ++i) {
//        new_data[i] = _data[i];
//      }
//    }
//    delete[] _data;
//    _data = new_data;
//    _capacity = n;
//  }
//}
//
template <class T>
HOSTDEV constexpr void
Vector<T>::resize(Size const n) noexcept
{
  Size const cs = size();
  // If we are shrinking, destroy the elements that are no longer needed
  // If we are growing, default construct the new elements
  if (cs < n) {
    append_default(n - cs);
  } else if (cs > n) {
    destruct_at_end(_begin + n);
  }
}
//
// template <class T>
// HOSTDEV inline void Vector<T>::push_back(T const & value)
//{
//  reserve(_size + 1);
//  _data[_size++] = value;
//}
//
// template <class T>
// HOSTDEV void Vector<T>::insert(T const * pos, Size const n, T const & value)
//{
//  if (n == 0) {
//    return;
//  }
//  auto const offset = static_cast<Size>(pos - _data);
//  assert(0 <= offset && offset <= _size);
//  Size const new_size = _size + n;
//  reserve(new_size);
//  // Shift elements to make room for the insertion
//  for (Size i = _size - 1; i >= offset; --i) {
//    _data[i + n] = _data[i];
//  }
//  // Insert elements
//  for (Size i = offset; i < offset + n; ++i) {
//    _data[i] = value;
//  }
//  _size = new_size;
//}
//
// template <class T>
// HOSTDEV void Vector<T>::insert(T const * pos, T const & value)
//{
//  insert(pos, 1, value);
//}
//
// template <class T>
// PURE HOSTDEV constexpr auto Vector<T>::contains(T const & value) const noexcept
//    -> bool requires(!std::floating_point<T>)
//{
//  for (Size i = 0; i < _size; ++i) {
//    if (_data[i] == value) {
//      return true;
//    }
//  }
//  return false;
//}
//
//// A classic abs(a - b) <= epsilon comparison
// template <class T>
// requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T const epsilon) noexcept -> bool
//{
//   if (a.size() != b.size()) {
//     return false;
//   }
//   for (Size i = 0; i < a.size(); ++i) {
//     if (std::abs(a[i] - b[i]) > epsilon) {
//       return false;
//     }
//   }
//   return true;
// }
//
// template <class T>
// requires(std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T const epsilon) noexcept -> bool
//{
//   if (a.size() != b.size()) {
//     return false;
//   }
//   for (Size i = 0; i < a.size(); ++i) {
//     T const diff = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
//     if (diff > epsilon) {
//       return false;
//     }
//   };
//   return true;
// }

// ----------------------------------------------------------------------------
// Hidden
// ----------------------------------------------------------------------------

template <class T>
HOSTDEV constexpr void
Vector<T>::allocate(Size n) noexcept
{
  assert(n < max_size());
  assert(_begin == nullptr);
  _begin = static_cast<T *>(::operator new(static_cast<size_t>(n) * sizeof(T)));
  _end = _begin;
  _end_cap = _begin + n;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::construct_at_end(Size n) noexcept
{
  Ptr new_end = _end + n;
  for (Ptr pos = _end; pos != new_end; ++pos) {
    construct_at(pos);
  }
  _end = new_end;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::construct_at_end(Size n, T const & value) noexcept
{
  Ptr new_end = _end + n;
  for (Ptr pos = _end; pos != new_end; ++pos) {
    construct_at(pos, value);
  }
  _end = new_end;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::destruct_at_end(Ptr new_last) noexcept
{
  Ptr soon_to_be_end = _end;
  while (new_last != soon_to_be_end) {
    destroy_at(--soon_to_be_end);
  }
  _end = new_last;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::append_default(Size n) noexcept
{
  // If we have enough capacity, just construct the new elements
  if (static_cast<Size>(_end_cap - _end) >= n) {
    construct_at_end(n);
  } else { // Otherwise, allocate a new buffer and move the elements over
    Size const current_size = size();
    Size const new_size = current_size + n;
    Size const new_capacity = recommend(new_size);
    Ptr new_begin =
        static_cast<T *>(::operator new(static_cast<size_t>(new_capacity) * sizeof(T)));
    Ptr new_end = new_begin;
    // Move the elements over
    for (Ptr old_pos = _begin; old_pos != _end; ++old_pos, ++new_end) {
      construct_at(new_end, move(*old_pos));
    }
    // Destroy the old elements
    destruct_at_end(_begin);
    // Update the pointers
    _begin = new_begin;
    _end = new_end;
    _end_cap = _begin + new_capacity;
    // Construct the new elements
    construct_at_end(n);
  }
}

template <class T>
HOSTDEV constexpr auto
Vector<T>::recommend(Size new_size) const noexcept -> Size
{
  return thrust::max(2 * capacity(), new_size);
}

} // namespace um2
