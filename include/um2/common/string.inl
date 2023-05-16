namespace um2
{

//// -- Accessors --
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char & String::operator [] (len_t const i) {
//    UM2_ASSERT(0 <= i && i < this->_size);
//    return this->_data[i];
//}
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char const & String::operator [] (len_t const i) const {
//    UM2_ASSERT(0 <= i && i < this->_size);
//    return this->_data[i];
//}
//
UM2_PURE UM2_HOSTDEV constexpr char8_t * String::begin() const { return this->_data; }

UM2_PURE UM2_HOSTDEV constexpr char8_t * String::end() const { return this->_data + this->_size; }

UM2_PURE UM2_HOSTDEV constexpr char8_t const * String::cbegin() const { return this->_data; }

UM2_PURE UM2_HOSTDEV constexpr char8_t const * String::cend() const
{
  return this->_data + this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr len_t String::size() const { return this->_size; }

UM2_PURE UM2_HOSTDEV constexpr len_t String::capacity() const { return this->_capacity; }

UM2_PURE UM2_HOSTDEV constexpr char8_t * String::data() { return this->_data; }

UM2_PURE UM2_HOSTDEV constexpr char8_t const * String::data() const { return this->_data; }
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char & String::front()
//{
//    UM2_ASSERT(this->_size > 0);
//    return this->_data[0];
//}
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char const & String::front() const
//{
//    UM2_ASSERT(this->_size > 0);
//    return this->_data[0];
//}
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char & String::back()
//{
//    UM2_ASSERT(this->_size > 0);
//    return this->_data[this->_size - 1];
//}
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char const & String::back() const
//{
//    UM2_ASSERT(this->_size > 0);
//    return this->_data[this->_size - 1];
//}
//
//// -- Member functions --
//
// UM2_PURE UM2_HOSTDEV constexpr
// bool String::empty() const { return this->_size == 0; }
//
// UM2_PURE UM2_HOSTDEV constexpr
// bool String::contains(char const value) const
//{
//    for (len_t i = 0; i < this->_size; ++i) {
//        if (this->_data[i] == value) {
//            return true;
//        }
//    }
//    return false;
//}
//
// -- Constructors --

template <size_t N>
UM2_HOSTDEV String::String(char const (&s)[N])
{
  this->_size = N - 1;
  this->_capacity = N;
  this->_data = new char8_t[N];
  memcpy(this->_data, s, N);
}

UM2_HOSTDEV String::String(String const & s)
{
  this->_size = s._size;
  this->_capacity = s._capacity;
  this->_data = new char8_t[this->_capacity];
  memcpy(this->_data, s._data, static_cast<size_t>(this->_size + 1));
}

String::String(std::string const & s)
{
  this->_size = static_cast<len_t>(s.size());
  this->_capacity = static_cast<len_t>(bit_ceil(s.size() + 1));
  this->_data = new char8_t[this->_capacity];
  memcpy(this->_data, s.data(), static_cast<size_t>(this->_size + 1));
}

// -- Operators --

UM2_HOSTDEV String & String::operator=(String const & s)
{
  if (this != &s) {
    if (this->_capacity < s._size + 1) {
      delete[] this->_data;
      this->_capacity = bit_ceil(s._size + 1);
      this->_data = new char8_t[this->_capacity];
    }
    this->_size = s._size;
    memcpy(this->_data, s._data, static_cast<size_t>(this->_size + 1));
  }
  return *this;
}

template <size_t N>
UM2_HOSTDEV String & String::operator=(char const (&s)[N])
{
  if (this->_capacity < static_cast<len_t>(N)) {
    delete[] this->_data;
    this->_capacity = static_cast<len_t>(bit_ceil(N));
    this->_data = new char8_t[static_cast<size_t>(this->_capacity)];
  }
  this->_size = static_cast<len_t>(N - 1);
  memcpy(this->_data, s, N);
  return *this;
}

String & String::operator=(std::string const & s)
{
  if (this->_capacity < static_cast<len_t>(s.size() + 1)) {
    delete[] this->_data;
    this->_capacity = static_cast<len_t>(bit_ceil(s.size() + 1));
    this->_data = new char8_t[this->_capacity];
  }
  this->_size = static_cast<len_t>(s.size());
  memcpy(this->_data, s.data(), s.size() + 1);
  return *this;
}

UM2_PURE UM2_HOSTDEV constexpr bool String::operator==(String const & s) const
{
  if (this->_size != s._size) {
    return false;
  }
  for (len_t i = 0; i < this->_size; ++i) {
    if (this->_data[i] != s._data[i]) {
      return false;
    }
  }
  return true;
}

// UM2_PURE UM2_HOSTDEV constexpr bool String::operator < (String const & v) const
//{
//    // Lexicographical comparison.
//    std::string_view const s1(this->_data, static_cast<size_t>(this->_size));
//    std::string_view const s2(v._data, static_cast<size_t>(v._size));
//    return s1 < s2;
//}
//
// template <size_t N>
// UM2_PURE UM2_HOSTDEV constexpr bool String::operator == (char const (&s)[N]) const
//{
//    if (this->_size != static_cast<len_t>(N - 1)) { return false; }
//    for (len_t i = 0; i < this->_size; ++i) {
//        if (this->_data[i] != s[i]) { return false; }
//    }
//    return true;
//}
//
// UM2_PURE constexpr bool String::operator == (std::string const & s) const
//{
//    if (this->_size != static_cast<len_t>(s.size())) { return false; }
//    for (len_t i = 0; i < this->_size; ++i) {
//        if (this->_data[i] != s[static_cast<size_t>(i)]) { return false; }
//    }
//    return true;
//}
//
//// -- Methods --
//
// UM2_PURE constexpr bool String::starts_with(std::string const & s) const
//{
//    if (this->_size < static_cast<len_t>(s.size())) { return false; }
//    for (len_t i = 0; i < static_cast<len_t>(s.size()); ++i) {
//        if (this->_data[i] != s[static_cast<size_t>(i)]) { return false; }
//    }
//    return true;
//}
//
// UM2_PURE constexpr bool String::ends_with(std::string const & s) const
//{
//    len_t const ssize = static_cast<len_t>(s.size());
//    len_t const vsize = this->_size;
//    if (vsize < ssize) { return false; }
//    for (len_t i = 0; i < ssize; ++i) {
//        if (this->_data[vsize - 1 - i] != s[s.size() - 1 - static_cast<size_t>(i)]) {
//            return false;
//        }
//    }
//    return true;
//}

} // namespace um2
