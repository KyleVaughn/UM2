add_library(um2 um2c.cpp ${UM2_HEADERS})
target_include_directories(um2 PUBLIC ${UM2_INCLUDE_DIR}) 
target_link_libraries(um2 PRIVATE Thrust)

if (UM2_ENABLE_CLANG_TIDY)
  set_target_properties(um2 PROPERTIES CXX_CLANG_TIDY "clang-tidy;--header-filter=${UM2_INCLUDE_DIR}")
endif()
