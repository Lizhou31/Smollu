# Smollu VM Component
cmake_minimum_required(VERSION 3.15)

# VM Library
add_library(smollu_vm STATIC smollu_vm.c)

target_include_directories(smollu_vm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# VM Tests
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_STREAMING_TEST "Build streaming VM test" ON)

if(BUILD_TESTS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CRITERION REQUIRED criterion)

    # VM unit tests
    add_executable(test_smollu_vm
        test/test_smollu_vm.c
    )

    target_include_directories(test_smollu_vm PRIVATE
        ${CRITERION_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(test_smollu_vm
        smollu_vm
        ${CRITERION_LIBRARIES}
    )

    target_compile_options(test_smollu_vm PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_vm PRIVATE ${CRITERION_LIBRARY_DIRS})

    # Test is registered at the top level in cmake/tests.cmake
endif()

# Streaming VM test (standalone executable)
if(BUILD_STREAMING_TEST)
    add_executable(test_streaming_vm
        test_streaming.c
        smollu_vm.c
    )

    target_include_directories(test_streaming_vm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_compile_definitions(test_streaming_vm PRIVATE
        CONFIG_SMOLLU_STREAMING=1
    )

    target_link_libraries(test_streaming_vm
        m  # Math library
    )

    # Copy demo bytecode for testing
    add_custom_command(TARGET test_streaming_vm POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/vm/test_data
        COMMENT "Creating test data directory"
    )
endif()