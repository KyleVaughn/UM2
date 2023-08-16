# code coverage installation

## enable gcov

1. turn on the cmake option `UM2_ENABLE_COVERAGE`, this option is synchronized with the `ENABLE_DEN_MOdE` option of
   the `cmake` command.
    1. in `cmake/add_um2_test.cmake`
    ```cmake
    if(UM2_ENABLE_COVERAGE)
    target_compile_options(${target} PRIVATE -fprofile-arcs -ftest-coverage)
    target_link_libraries(${target} PRIVATE gcov)
    ```

    2. in `cmake/um2_config.cmake`
    ```cmake
    if (UM2_ENABLE_COVERAGE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")
    target_link_libraries(um2 PUBLIC gcov)
    endif ()
    ```

2. the github action will automatically generate the `coverage` report, you can find it in the `Actions` tab of the
   github repository and will upload it to coverall.io.
    