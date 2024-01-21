set(
  FORMAT_PATTERNS
  src/*.cpp
  include/*.hpp
  include/*.h
  tests/*.hpp
  tests/*.cpp
  examples/*.hpp
  examples/*.cpp
  benchmarks/*.hpp
  benchmarks/*.cpp
  tutorial/*.hpp
  tutorial/*.cpp
  CACHE STRING
  "Patterns to format")

find_program(CLANG_FORMAT clang-format)
if (NOT CLANG_FORMAT)
  message(FATAL_ERROR "Could not find clang-format")
endif()
set(FORMAT_COMMAND ${CLANG_FORMAT})

add_custom_target(
  format-check
  COMMAND "${CMAKE_COMMAND}"
  -D "FORMAT_COMMAND=${FORMAT_COMMAND}"
  -D "PATTERNS=${FORMAT_PATTERNS}"
  -P "${PROJECT_SOURCE_DIR}/cmake/format.cmake"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Formatting the code"
  VERBATIM)

add_custom_target(
  format-fix
  COMMAND "${CMAKE_COMMAND}"
  -D "FORMAT_COMMAND=${FORMAT_COMMAND}"
  -D "PATTERNS=${FORMAT_PATTERNS}"
  -D FIX=YES
  -P "${PROJECT_SOURCE_DIR}/cmake/format.cmake"
  WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  COMMENT "Fixing the code"
  VERBATIM)
