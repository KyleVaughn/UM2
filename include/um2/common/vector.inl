namespace um2
{

//// -- Accessors --
//
// template <typename T>
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// T & Vector<T>::operator [] (len_t const i) {
//    UM2_ASSERT(0 <= i && i < this->size_);
//    return this->_data[i];
//}
//
// template <typename T>
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// T const & Vector<T>::operator [] (len_t const i) const {
//    UM2_ASSERT(0 <= i && i < this->size_);
//    return this->_data[i];
//}
//
// template <typename T>
// UM2_PURE UM2_HOSTDEV constexpr
// T * Vector<T>::begin() const { return this->_data; }
//
// template <typename T>
// UM2_PURE UM2_HOSTDEV constexpr
// T * Vector<T>::end() const { return this->_data + this->size_; }
//
// template <typename T>
// UM2_PURE UM2_HOSTDEV constexpr
// T const * Vector<T>::cbegin() const { return this->_data; }
//
// template <typename T>
// UM2_PURE UM2_HOSTDEV constexpr
// T const * Vector<T>::cend() const { return this->_data + this->size_; }
//
template <typename T> UM2_PURE UM2_HOSTDEV constexpr len_t Vector<T>::size() const
{
  return this->size_;
}

template <typename T> UM2_PURE UM2_HOSTDEV constexpr len_t Vector<T>::capacity() const
{
  return this->_capacity;
}

template <typename T> UM2_PURE UM2_HOSTDEV constexpr T * Vector<T>::data() { return this->_data; }
//
// template <typename T>
// UM2_PURE UM2_HOSTDEV constexpr
// T const * Vector<T>::data() const { return this->_data; }
//
// template <typename T>
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// T & Vector<T>::front()
//{
//    UM2_ASSERT(this->size_ > 0);
//    return this->_data[0];
//}
//
// template <typename T>
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// T const & Vector<T>::front() const
//{
//    UM2_ASSERT(this->size_ > 0);
//    return this->_data[0];
//}
//
// template <typename T>
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// T & Vector<T>::back()
//{
//    UM2_ASSERT(this->size_ > 0);
//    return this->_data[this->size_ - 1];
//}
//
// template <typename T>
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// T const & Vector<T>::back() const
//{
//    UM2_ASSERT(this->size_ > 0);
//    return this->_data[this->size_ - 1];
//}
//
//// -- Member functions --
//
template <typename T> UM2_HOSTDEV void Vector<T>::clear()
{
  this->size_ = 0;
  this->_capacity = 0;
  delete[] this->_data;
  this->_data = nullptr;
}
//
// template <typename T>
// UM2_HOSTDEV inline
// void Vector<T>::reserve(len_t n) {
//    if (this->_capacity < n) {
//        // since n > 0, we are safe to cast to unsigned and use bit_ceil
//        // to determine the next power of 2
//        n = static_cast<len_t>(bit_ceil(n));
//        T * new_data = new T[static_cast<size_t>(n)];
//        for (len_t i = 0; i < this->size_; ++i) {
//            new_data[i] = this->_data[i];
//        }
//        delete[] this->_data;
//        this->_data = new_data;
//        this->_capacity = n;
//    }
//}
//
// template <typename T>
// UM2_HOSTDEV
// void Vector<T>::resize(len_t n) {
//    this->reserve(n);
//    this->size_ = n;
//}
//
// template <typename T>
// UM2_HOSTDEV
// inline void Vector<T>::push_back(T const & value) {
//    this->reserve(this->size_ + 1);
//    this->_data[this->size_++] = value;
//}
//
// template <typename T>
// UM2_PURE UM2_HOSTDEV constexpr
// bool Vector<T>::empty() const { return this->size_ == 0; }
//
// template <typename T>
// UM2_HOSTDEV
// void Vector<T>::insert(T const * pos, len_t const n, T const & value)
//{
//    if (n == 0) { return; }
//    len_t const offset = static_cast<len_t>(pos - this->_data);
//    UM2_ASSERT(0 <= offset && offset <= this->size_);
//    len_t const newsize_ = this->size_ + n;
//    this->reserve(newsize_);
//    // Shift elements to make room for the insertion
//    for (len_t i = this->size_ - 1; i >= offset; --i)
//    {
//        this->_data[i + n] = this->_data[i];
//    }
//    // Insert elements
//    for (len_t i = offset; i < offset + n; ++i)
//    {
//        this->_data[i] = value;
//    }
//    this->size_ = newsize_;
//}
//
// template <typename T>
// UM2_HOSTDEV
// void Vector<T>::insert(T const * pos, T const & value)
//{
//    this->insert(pos, 1, value);
//}
//
// template <typename T>
// UM2_PURE UM2_HOSTDEV constexpr
// bool Vector<T>::contains(T const & value) const
//{
//    for (len_t i = 0; i < this->size_; ++i) {
//        if (this->_data[i] == value) {
//            return true;
//        }
//    }
//    return false;
//}
//
//// -- Constructors --
//
template <typename T>
UM2_HOSTDEV Vector<T>::Vector(len_t const n)
    : size_{n}, _capacity{n}, _data{new T[static_cast<size_t>(n)]}
{
  assert(n > 0);
}

template <typename T>
UM2_HOSTDEV Vector<T>::Vector(len_t const n, T const & value)
    : size_{n}, _capacity{n}, _data{new T[static_cast<size_t>(n)]}
{
  assert(n > 0);
  for (len_t i = 0; i < n; ++i) {
    this->_data[i] = value;
  }
}

// template <typename T>
// UM2_HOSTDEV
// Vector<T>::Vector(Vector<T> const & v) :
//     size_{v.size_},
//     _capacity{static_cast<len_t>(bit_ceil(v.size_))},
//     _data{new T[static_cast<size_t>(this->_capacity)]}
//{
//     for (len_t i = 0; i < v.size_; ++i) {
//         this->_data[i] = v._data[i];
//     }
// }
//
// template <typename T>
// UM2_HOSTDEV
// Vector<T>::Vector(std::initializer_list<T> const & list) :
//     size_{static_cast<len_t>(list.size())},
//     _capacity{static_cast<len_t>(bit_ceil(list.size()))},
//     _data{new T[static_cast<size_t>(this->_capacity)]}
//{
//     len_t i = 0;
//     for (auto const & value : list) {
//         this->_data[i++] = value;
//     }
// }
//
//// -- Operators --
//
// template <typename T>
// UM2_HOSTDEV
// Vector<T> & Vector<T>::operator = (Vector<T> const & v) {
//    if (this != &v) {
//        if (this->_capacity < v.size()) {
//            delete[] this->_data;
//            this->_data = new T[static_cast<size_t>(bit_ceil(v.size()))];
//            this->_capacity = bit_ceil(v.size());
//        }
//        this->size_ = v.size();
//        for (len_t i = 0; i < v.size(); ++i) {
//            this->_data[i] = v._data[i];
//        }
//    }
//    return *this;
//}
//
// template <typename T>
// UM2_PURE UM2_HOSTDEV constexpr
// bool Vector<T>::operator == (Vector<T> const & v) const {
//    if (this->size_ != v.size_) { return false; }
//    for (len_t i = 0; i < this->size_; ++i) {
//        if (this->_data[i] != v._data[i]) { return false; }
//    }
//    return true;
//}
//
//// -- IO --
//
// template <typename T>
// std::ostream & operator << (std::ostream & os, Vector<T> const & v) {
//    os << '(';
//    for (len_t i = 0; i < v.size(); ++i) {
//        os << v[i];
//        if (i < v.size() - 1) os << ", ";
//    }
//    os << ')';
//    return os;
//}
//
//// -- Methods --
//
//// A classic abs(a - b) <= epsilon comparison
// template <typename T>
// UM2_NDEBUG_PURE UM2_HOST constexpr
// bool is_approx(Vector<T> const & a, Vector<T> const & b, T const & epsilon)
//{
//     if (a.size() != b.size()) { return false; }
//     struct approx_functor {
//
//         typedef thrust::tuple<T, T> Tuple;
//
//         T const epsilon;
//
//         UM2_PURE UM2_HOSTDEV constexpr
//         bool operator()(Tuple const & tuple) const {
//             T const & a = thrust::get<0>(tuple);
//             T const & b = thrust::get<1>(tuple);
//             return std::abs(a - b) <= epsilon;
//         }
//     };
//
//     return thrust::all_of(
//             thrust::make_zip_iterator(thrust::make_tuple(a.cbegin(), b.cbegin())),
//             thrust::make_zip_iterator(thrust::make_tuple(a.cend(), b.cend())),
//             approx_functor{epsilon});
// }
//
//#if UM2_HAS_CUDA
// template <typename T>
// UM2_NDEBUG_PURE UM2_DEVICE constexpr
// bool is_approx(Vector<T> const & a, Vector<T> const & b, T const & epsilon)
//{
//     if (a.size() != b.size()) { return false; }
//     struct approx_functor {
//
//         typedef thrust::tuple<T, T> Tuple;
//
//         T const epsilon;
//
//         UM2_PURE UM2_HOSTDEV constexpr
//         bool operator()(Tuple const & tuple) const {
//             T const & a = thrust::get<0>(tuple);
//             T const & b = thrust::get<1>(tuple);
//             return std::abs(a - b) <= epsilon;
//         }
//     };
//
//     return thrust::all_of(
//             thrust::seq,
//             thrust::make_zip_iterator(thrust::make_tuple(a.cbegin(), b.cbegin())),
//             thrust::make_zip_iterator(thrust::make_tuple(a.cend(), b.cend())),
//             approx_functor{epsilon});
// }
//#endif

} // namespace um2
