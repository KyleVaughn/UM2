namespace um2
{

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::size() const noexcept -> len_t
{
  return this->_size;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::capacity() const noexcept -> len_t
{
  return this->_capacity;
}

template <typename T>
// cppcheck-suppress functionConst
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::data() noexcept -> T *
{
  return this->_data;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::data() const noexcept -> T const *
{
  return this->_data;
}

template <typename T>
// cppcheck-suppress functionConst
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::begin() noexcept -> T *
{
  return this->_data;
}

template <typename T>
// cppcheck-suppress functionConst
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::end() noexcept -> T *
{
  return this->_data + this->_size;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::cbegin() const noexcept -> T const *
{
  return this->_data;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::cend() const noexcept -> T const *
{
  return this->_data + this->_size;
}

template <typename T>
// cppcheck-suppress functionConst
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::front() noexcept -> T &
{
  return this->_data[0];
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::front() const noexcept -> T const &
{
  return this->_data[0];
}

template <typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto Vector<T>::back() noexcept -> T &
{
  assert(this->_size > 0);
  return this->_data[this->_size - 1];
}

template <typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto Vector<T>::back() const noexcept -> T const &
{
  assert(this->_size > 0);
  return this->_data[this->_size - 1];
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV Vector<T>::Vector(len_t const n)
    : _size{n}, _capacity{static_cast<len_t>(bit_ceil(n))}, _data{new T[bit_ceil(n)]}
{
  assert(n > 0);
}

template <typename T>
UM2_HOSTDEV Vector<T>::Vector(len_t const n, T const & value)
    : _size{n}, _capacity{static_cast<len_t>(bit_ceil(n))}, _data{new T[bit_ceil(n)]}
{
  assert(n > 0);
  // Trusting the compiler to optimize this to memset for appropriate types.
  for (len_t i = 0; i < n; ++i) {
    this->_data[i] = value;
  }
}

template <typename T>
UM2_HOSTDEV Vector<T>::Vector(Vector<T> const & v)
    : _size{v._size}, _capacity{static_cast<len_t>(bit_ceil(v._size))},
      _data{new T[bit_ceil(v._size)]}
{

  if constexpr (std::is_trivially_copyable_v<T>) {
    memcpy(this->_data, v._data, static_cast<size_t>(v._size) * sizeof(T));
  } else {
    for (len_t i = 0; i < v._size; ++i) {
      this->_data[i] = v._data[i];
    }
  }
}

template <typename T>
UM2_HOSTDEV Vector<T>::Vector(Vector<T> && v) noexcept
    : _size{v._size}, _capacity{v._capacity}, _data{v._data}
{
  v._size = 0;
  v._capacity = 0;
  v._data = nullptr;
}

template <typename T>
Vector<T>::Vector(std::initializer_list<T> const & list)
    : _size{static_cast<len_t>(list.size())}, _capacity{static_cast<len_t>(
                                                  bit_ceil(list.size()))},
      _data{new T[bit_ceil(list.size())]}
{
  if constexpr (std::is_trivially_copyable_v<T>) {
    memcpy(this->_data, list.begin(), list.size() * sizeof(T));
  } else {
    len_t i = 0;
    for (auto const & value : list) {
      this->_data[i++] = value;
    }
  }
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

template <typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto Vector<T>::operator[](len_t const i) noexcept
    -> T &
{
  assert(0 <= i && i < this->_size);
  return this->_data[i];
}

template <typename T>
UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
Vector<T>::operator[](len_t const i) const noexcept -> T const &
{
  assert(0 <= i && i < this->_size);
  return this->_data[i];
}

template <typename T>
UM2_HOSTDEV auto Vector<T>::operator=(Vector<T> const & v) -> Vector<T> &
{
  if (this != &v) {
    if (this->_capacity < v.size()) {
      delete[] this->_data;
      this->_data = new T[bit_ceil(v.size())];
      this->_capacity = static_cast<len_t>(bit_ceil(v.size()));
    }
    this->_size = v.size();
    if constexpr (std::is_trivially_copyable_v<T>) {
      memcpy(this->_data, v._data, static_cast<size_t>(v.size()) * sizeof(T));
    } else {
      for (len_t i = 0; i < v.size(); ++i) {
        this->_data[i] = v._data[i];
      }
    }
  }
  return *this;
}

template <typename T>
UM2_HOSTDEV auto Vector<T>::operator=(Vector<T> && v) noexcept -> Vector<T> &
{
  if (this != &v) {
    delete[] this->_data;
    this->_size = v._size;
    this->_capacity = v._capacity;
    this->_data = v._data;
    v._size = 0;
    v._capacity = 0;
    v._data = nullptr;
  }
  return *this;
}

#ifdef __CUDA_ARCH__
template <typename T>
UM2_PURE __device__ constexpr auto
Vector<T>::operator==(Vector<T> const & v) const noexcept -> bool
{
  if (this->_size != v._size) {
    return false;
  }
  for (len_t i = 0; i < this->_size; ++i) {
    if (this->_data[i] != v._data[i]) {
      return false;
    }
  }
  return true;
}
#else
template <typename T>
UM2_PURE UM2_HOST constexpr auto Vector<T>::operator==(Vector<T> const & v) const noexcept
    -> bool
{
  if (this->_size != v._size) {
    return false;
  }
  if constexpr (std::is_trivially_copyable_v<T>) {
    return memcmp(this->_data, v._data, static_cast<size_t>(this->_size) * sizeof(T)) ==
           0;
  } else {
    for (len_t i = 0; i < this->_size; ++i) {
      if (this->_data[i] != v._data[i]) {
        return false;
      }
    }
  }
  return true;
}
#endif

// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV void Vector<T>::clear() noexcept
{
  this->_size = 0;
  this->_capacity = 0;
  delete[] this->_data;
  this->_data = nullptr;
}

template <typename T>
UM2_HOSTDEV inline void Vector<T>::reserve(len_t n)
{
  if (this->_capacity < n) {
    // since n > 0, we are safe to cast to unsigned and use bit_ceil
    // to determine the next power of 2
    n = static_cast<len_t>(bit_ceil(n));
    T * new_data = new T[static_cast<size_t>(n)];
    if constexpr (std::is_trivially_copyable_v<T>) {
      memcpy(new_data, this->_data, static_cast<size_t>(this->_size) * sizeof(T));
    } else {
      for (len_t i = 0; i < this->_size; ++i) {
        new_data[i] = this->_data[i];
      }
    }
    delete[] this->_data;
    this->_data = new_data;
    this->_capacity = n;
  }
}

template <typename T>
UM2_HOSTDEV void Vector<T>::resize(len_t const n)
{
  this->reserve(n);
  this->_size = n;
}

template <typename T>
UM2_HOSTDEV inline void Vector<T>::push_back(T const & value)
{
  this->reserve(this->_size + 1);
  this->_data[this->_size++] = value;
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::empty() const -> bool
{
  return this->_size == 0;
}

template <typename T>
UM2_HOSTDEV void Vector<T>::insert(T const * pos, len_t const n, T const & value)
{
  if (n == 0) {
    return;
  }
  auto const offset = static_cast<len_t>(pos - this->_data);
  assert(0 <= offset && offset <= this->_size);
  len_t const new_size = this->_size + n;
  this->reserve(new_size);
  // Shift elements to make room for the insertion
  for (len_t i = this->_size - 1; i >= offset; --i) {
    this->_data[i + n] = this->_data[i];
  }
  // Insert elements
  for (len_t i = offset; i < offset + n; ++i) {
    this->_data[i] = value;
  }
  this->_size = new_size;
}

template <typename T>
UM2_HOSTDEV void Vector<T>::insert(T const * pos, T const & value)
{
  this->insert(pos, 1, value);
}

template <typename T>
UM2_PURE UM2_HOSTDEV constexpr auto Vector<T>::contains(T const & value) const noexcept
    -> bool requires(!std::floating_point<T>)
{
  for (len_t i = 0; i < this->_size; ++i) {
    if (this->_data[i] == value) {
      return true;
    }
  }
  return false;
}

// A classic abs(a - b) <= epsilon comparison
template <typename T>
requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) UM2_PURE UM2_HOSTDEV
    constexpr auto is_approx(Vector<T> const & a, Vector<T> const & b,
                             T const epsilon) noexcept -> bool
{
  if (a.size() != b.size()) {
    return false;
  }
  struct ApproxFunctor {

    T epsilon;

    UM2_PURE UM2_HOSTDEV constexpr auto
    operator()(thrust::tuple<T, T> const & tuple) const -> bool
    {
      return std::abs(thrust::get<0>(tuple) - thrust::get<1>(tuple)) <= epsilon;
    }
  };

  return thrust::all_of(
      thrust::seq, thrust::make_zip_iterator(thrust::make_tuple(a.cbegin(), b.cbegin())),
      thrust::make_zip_iterator(thrust::make_tuple(a.cend(), b.cend())),
      ApproxFunctor{epsilon});
}

template <typename T>
requires(std::unsigned_integral<T>) UM2_PURE UM2_HOSTDEV
    constexpr auto is_approx(Vector<T> const & a, Vector<T> const & b,
                             T const epsilon) noexcept -> bool
{
  if (a.size() != b.size()) {
    return false;
  }
  struct ApproxFunctor {

    T const epsilon;

    UM2_PURE UM2_HOSTDEV constexpr auto
    operator()(thrust::tuple<T, T> const & tuple) const -> bool
    {
      T const a = thrust::get<0>(tuple);
      T const b = thrust::get<1>(tuple);
      T const diff = a > b ? a - b : b - a;
      return diff <= epsilon;
    }
  };

  return thrust::all_of(
      thrust::seq, thrust::make_zip_iterator(thrust::make_tuple(a.cbegin(), b.cbegin())),
      thrust::make_zip_iterator(thrust::make_tuple(a.cend(), b.cend())),
      ApproxFunctor{epsilon});
}

} // namespace um2
