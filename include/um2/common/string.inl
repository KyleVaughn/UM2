namespace um2
{

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

UM2_PURE UM2_HOSTDEV constexpr auto String::begin() const -> char8_t *
{
  return this->_data;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::end() const -> char8_t *
{
  return this->_data + this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::cbegin() const -> char8_t const *
{
  return this->_data;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::cend() const -> char8_t const *
{
  return this->_data + this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::size() const -> len_t { return this->_size; }

UM2_PURE UM2_HOSTDEV constexpr auto String::capacity() const -> len_t
{
  return this->_capacity;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::data() -> char8_t * { return this->_data; }

UM2_PURE UM2_HOSTDEV constexpr auto String::data() const -> char8_t const *
{
  return this->_data;
}

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

template <size_t N>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
UM2_HOSTDEV String::String(char const (&s)[N])
    : _size(N - 1), _capacity(N), _data(new char8_t[N])
{
  memcpy(this->_data, s, N);
}

// --------------------------------------------------------------------------
// Operators
// --------------------------------------------------------------------------

template <size_t N>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
UM2_HOSTDEV auto String::operator=(char const (&s)[N]) -> String &
{
  if (this->_capacity < static_cast<len_t>(N)) {
    delete[] this->_data;
    this->_capacity = static_cast<len_t>(bit_ceil(N));
    this->_data = new char8_t[bit_ceil(N)];
  }
  this->_size = static_cast<len_t>(N - 1);
  memcpy(this->_data, s, N);
  return *this;
}

template <size_t N>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
UM2_PURE UM2_HOSTDEV auto String::operator==(char const (&s)[N]) const -> bool
{
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return this->compare(reinterpret_cast<char8_t const *>(s)) == 0;
}

UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto String::operator[](len_t const i) -> char8_t &
{
  assert(0 <= i && i < this->_size);
  return this->_data[i];
}

UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto String::operator[](len_t const i) const
    -> char8_t const &
{
  assert(0 <= i && i < this->_size);
  return this->_data[i];
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

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
