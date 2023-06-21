namespace um2
{

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

UM2_PURE UM2_HOSTDEV constexpr auto
String::begin() const noexcept -> char *
{
  return this->_data;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::end() const noexcept -> char *
{
  return this->_data + this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::cbegin() const noexcept -> char const *
{
  return this->_data;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::cend() const noexcept -> char const *
{
  return this->_data + this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::size() const noexcept -> len_t
{
  return this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::capacity() const noexcept -> len_t
{
  return this->_capacity;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::data() noexcept -> char *
{
  return this->_data;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::data() const noexcept -> char const *
{
  return this->_data;
}

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

template <size_t N>
UM2_HOSTDEV
String::String(char const (&s)[N])
    : _size(N - 1),
      _capacity(static_cast<len_t>(bit_ceil(N))),
      _data(new char[bit_ceil(N)])
{
  memcpy(this->_data, s, N);
}

// --------------------------------------------------------------------------
// Operators
// --------------------------------------------------------------------------

template <size_t N>
UM2_HOSTDEV auto
String::operator=(char const (&s)[N]) -> String &
{
  if (this->_capacity < static_cast<len_t>(N)) {
    delete[] this->_data;
    this->_capacity = static_cast<len_t>(bit_ceil(N));
    this->_data = new char[bit_ceil(N)];
  }
  this->_size = static_cast<len_t>(N - 1);
  memcpy(this->_data, s, N);
  return *this;
}

template <size_t N>
// NOLINTNEXTLINE(*-avoid-c-arrays)
UM2_PURE UM2_HOSTDEV constexpr auto
String::operator==(char const (&s)[N]) const noexcept -> bool
{
  if (this->_size != static_cast<len_t>(N - 1)) {
    return false;
  }
  for (len_t i = 0; i < this->_size; ++i) {
    if (this->_data[i] != s[i]) {
      return false;
    }
  }
  return true;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::operator==(String const & s) const noexcept -> bool
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

UM2_PURE UM2_HOSTDEV constexpr auto
String::operator<(String const & s) const noexcept -> bool
{
  len_t const min_size = this->_size < s._size ? this->_size : s._size;
  for (len_t i = 0; i < min_size; ++i) {
    if (this->_data[i] != s._data[i]) {
      return this->_data[i] < s._data[i];
    }
  }
  return this->_size < s._size;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::operator<=(String const & s) const noexcept -> bool
{
  len_t const min_size = this->_size < s._size ? this->_size : s._size;
  for (len_t i = 0; i < min_size; ++i) {
    if (this->_data[i] != s._data[i]) {
      return this->_data[i] < s._data[i];
    }
  }
  return this->_size <= s._size;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::operator>(String const & s) const noexcept -> bool
{
  len_t const min_size = this->_size < s._size ? this->_size : s._size;
  for (len_t i = 0; i < min_size; ++i) {
    if (this->_data[i] != s._data[i]) {
      return this->_data[i] > s._data[i];
    }
  }
  return this->_size > s._size;
}

UM2_PURE UM2_HOSTDEV constexpr auto
String::operator>=(String const & s) const noexcept -> bool
{
  len_t const min_size = this->_size < s._size ? this->_size : s._size;
  for (len_t i = 0; i < min_size; ++i) {
    if (this->_data[i] != s._data[i]) {
      return this->_data[i] > s._data[i];
    }
  }
  return this->_size >= s._size;
}

UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
String::operator[](len_t const i) noexcept -> char &
{
  assert(0 <= i && i < this->_size);
  return this->_data[i];
}

UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
String::operator[](len_t const i) const noexcept -> char const &
{
  assert(0 <= i && i < this->_size);
  return this->_data[i];
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

UM2_PURE UM2_HOSTDEV constexpr auto
String::contains(char const c) const noexcept -> bool
{
  for (len_t i = 0; i < this->_size; ++i) {
    if (this->_data[i] == c) {
      return true;
    }
  }
  return false;
}

UM2_PURE constexpr auto
String::starts_with(std::string const & s) const noexcept -> bool
{

  if (this->_size < static_cast<len_t>(s.size())) {
    return false;
  }
  char const * const sdata = s.data();
  for (len_t i = 0; i < static_cast<len_t>(s.size()); ++i) {
    if (this->_data[i] != sdata[i]) {
      return false;
    }
  }
  return true;
}

UM2_PURE constexpr auto
String::ends_with(std::string const & s) const noexcept -> bool
{
  auto const ssize = static_cast<len_t>(s.size());
  len_t const vsize = this->_size;
  if (vsize < ssize) {
    return false;
  }
  char const * const sdata = s.data();
  for (len_t i = 0; i < ssize; ++i) {
    if (this->_data[vsize - 1 - i] != sdata[ssize - 1 - i]) {
      return false;
    }
  }
  return true;
}

} // namespace um2
