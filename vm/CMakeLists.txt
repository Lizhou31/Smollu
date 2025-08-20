# Smollu VM Component
cmake_minimum_required(VERSION 3.15)

# VM Library
add_library(smollu_vm STATIC smollu_vm.c)

target_include_directories(smollu_vm PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# VM Tests
option(BUILD_TESTS "Build unit tests" ON)

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