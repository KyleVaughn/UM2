namespace um2
{
// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Size const n)
    : _begin{new T[static_cast<uint64_t>(n)]},
      _end{_begin + n},
      _end_cap{_begin + n}
{
}

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Size const n, T const & value)
    : _begin{new T[static_cast<uint64_t>(n)]},
      _end{_begin + n},
      _end_cap{_begin + n}
{
  for (auto pos = _begin; pos != _end; ++pos) {
    *pos = value;
  }
}

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Vector<T> const & v)
    : _begin{new T[static_cast<uint64_t>(v.size())]},
      _end{_begin + v.size()},
      _end_cap{_begin + v.size()}
{
  copy(v.begin(), v.end(), _begin);
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
HOSTDEV constexpr Vector<T>::Vector(std::initializer_list<T> const & list)
    : _begin{new T[list.size()]},
      _end{_begin + list.size()},
      _end_cap{_begin + list.size()}
{
  copy(list.begin(), list.end(), _begin);
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

template <class T>
HOSTDEV constexpr Vector<T>::~Vector() noexcept
{
  delete[] _begin;
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

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
Vector<T>::operator=(Vector<T> const & v) -> Vector<T> &
{
  if (this != addressof(v)) {
    delete[] _begin;
    _begin = new T[static_cast<uint64_t>(v.size())];
    _end = _begin + v.size();
    _end_cap = _begin + v.size();
    copy(v.begin(), v.end(), _begin);
  }
  return *this;
}

template <class T>
HOSTDEV constexpr auto
Vector<T>::operator=(Vector<T> && v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    delete[] _begin;
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
  destroy(begin(), end());
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
// template <class T>
// HOSTDEV void Vector<T>::resize(Size const n)
//{
//  reserve(n);
//  _size = n;
//}
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

} // namespace um2
