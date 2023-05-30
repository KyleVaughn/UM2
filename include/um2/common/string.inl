namespace um2
{

// --------------------------------------------------------------------------
// Accessors
// --------------------------------------------------------------------------

UM2_PURE UM2_HOSTDEV constexpr auto String::begin() const noexcept -> char8_t *
{
  return this->_data;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::end() const noexcept -> char8_t *
{
  return this->_data + this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::cbegin() const noexcept -> char8_t const *
{
  return this->_data;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::cend() const noexcept -> char8_t const *
{
  return this->_data + this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::size() const noexcept -> len_t
{
  return this->_size;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::capacity() const noexcept -> len_t
{
  return this->_capacity;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::data() noexcept -> char8_t *
{
  return this->_data;
}

UM2_PURE UM2_HOSTDEV constexpr auto String::data() const noexcept -> char8_t const *
{
  return this->_data;
}

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

template <size_t N>
UM2_HOSTDEV String::String(char const (&s)[N])
    : _size(N - 1),
      _capacity(static_cast<len_t>(bit_ceil(N))),
      _data(new char8_t[bit_ceil(N)])
{
  memcpy(this->_data, s, N);
}

// --------------------------------------------------------------------------
// Operators
// --------------------------------------------------------------------------

template <size_t N>
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
// NOLINTNEXTLINE(*-avoid-c-arrays)
UM2_PURE UM2_HOSTDEV auto String::operator==(char const (&s)[N]) const noexcept -> bool
{
  return this->compare(reinterpret_cast<char8_t const *>(s)) == 0;
}

UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto String::operator[](len_t const i) noexcept
    -> char8_t &
{
  assert(0 <= i && i < this->_size);
  return this->_data[i];
}

UM2_NDEBUG_PURE UM2_HOSTDEV constexpr auto
String::operator[](len_t const i) const noexcept -> char8_t const &
{
  assert(0 <= i && i < this->_size);
  return this->_data[i];
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

UM2_PURE UM2_HOSTDEV constexpr auto String::contains(char const c) const noexcept -> bool
{
  auto const cc = static_cast<char8_t>(c);
  for (len_t i = 0; i < this->_size; ++i) {
    if (this->_data[i] == cc) {
      return true;
    }
  }
  return false;
}

UM2_PURE constexpr auto String::starts_with(std::string const & s) const noexcept -> bool
{

  if (this->_size < static_cast<len_t>(s.size())) {
    return false;
  }
  char8_t const * const data = this->_data;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto const * const sdata = reinterpret_cast<char8_t const *>(s.data());
  for (len_t i = 0; i < static_cast<len_t>(s.size()); ++i) {
    if (data[i] != sdata[i]) {
      return false;
    }
  }
  return true;
}

UM2_PURE constexpr auto String::ends_with(std::string const & s) const noexcept -> bool
{
  auto const ssize = static_cast<len_t>(s.size());
  len_t const vsize = this->_size;
  if (vsize < ssize) {
    return false;
  }
  char8_t const * const data = this->_data;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto const * const sdata = reinterpret_cast<char8_t const *>(s.data());
  for (len_t i = 0; i < ssize; ++i) {
    if (data[vsize - 1 - i] != sdata[ssize - 1 - i]) {
      return false;
    }
  }
  return true;
}

} // namespace um2
