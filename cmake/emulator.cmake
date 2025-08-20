# Rust Emulator integration with CMake

# Check if Rust/Cargo is available
find_program(CARGO_EXECUTABLE cargo)
if(NOT CARGO_EXECUTABLE)
    message(WARNING "Cargo not found. Rust emulator will not be built. Install Rust to enable emulator building.")
    return()
endif()

# Determine build type for Cargo
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CARGO_BUILD_TYPE "release")
    set(CARGO_BUILD_FLAGS "--release")
else()
    set(CARGO_BUILD_TYPE "debug")
    set(CARGO_BUILD_FLAGS "")
endif()

# Emulator source files to track for changes
file(GLOB_RECURSE EMULATOR_SOURCES 
    "${CMAKE_SOURCE_DIR}/emulator/src/*"
    "${CMAKE_SOURCE_DIR}/emulator/examples/*"
    "${CMAKE_SOURCE_DIR}/emulator/c_integration/*"
    "${CMAKE_SOURCE_DIR}/emulator/Cargo.toml"
    "${CMAKE_SOURCE_DIR}/emulator/build.rs"
)

# Custom target to build the Rust emulator
add_custom_target(smollu_emulator ALL
    COMMAND ${CARGO_EXECUTABLE} build ${CARGO_BUILD_FLAGS}
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/emulator"
    COMMENT "Building Rust emulator (${CARGO_BUILD_TYPE} mode)"
    SOURCES ${EMULATOR_SOURCES}
)

# The emulator depends on the C VM being built first (for FFI integration)
add_dependencies(smollu_emulator smollu_vm)

# Custom target to run emulator tests (when tests are enabled)
if(BUILD_TESTS)
    add_custom_target(test_emulator
        COMMAND ${CARGO_EXECUTABLE} test
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/emulator"
        COMMENT "Running Rust emulator tests"
        DEPENDS smollu_emulator
    )
    
    # Add emulator tests to CTest
    add_test(NAME emulator_tests 
        COMMAND ${CARGO_EXECUTABLE} test
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/emulator"
    )
    
    # Add emulator example tests to CTest
    add_test(NAME emulator_basic_test
        COMMAND ${CARGO_EXECUTABLE} run --example basic_test
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/emulator"
    )
    
    add_test(NAME emulator_simple_test
        COMMAND ${CARGO_EXECUTABLE} run --example simple_test  
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/emulator"
    )
endif()

# Custom target for emulator cleanup
add_custom_target(clean_emulator
    COMMAND ${CARGO_EXECUTABLE} clean
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/emulator"
    COMMENT "Cleaning Rust emulator build artifacts"
)

# Information about emulator executable location
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(EMULATOR_EXECUTABLE "${CMAKE_SOURCE_DIR}/emulator/target/release/smollu-emulator")
else()
    set(EMULATOR_EXECUTABLE "${CMAKE_SOURCE_DIR}/emulator/target/debug/smollu-emulator")
endif()

# Message about emulator integration
message(STATUS "Rust emulator integration enabled")
message(STATUS "  Build type: ${CARGO_BUILD_TYPE}")
message(STATUS "  Executable: ${EMULATOR_EXECUTABLE}")
message(STATUS "  Tests enabled: ${BUILD_TESTS}")