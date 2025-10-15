# cmake/llvm_from_submodule.cmake
# Configure LLVM/MLIR (static) from submodule and add as subdirectory

set(LLVM_SOURCE_DIR "${CMAKE_SOURCE_DIR}/external/llvm-project/llvm")
set(LLVM_BINARY_DIR "${CMAKE_BINARY_DIR}/llvm-build")

# Build MLIR (static), RTTI ON
# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default: Release)" FORCE)
endif()

set(LLVM_ENABLE_PROJECTS "mlir" CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD "Native;X86;AArch64" CACHE STRING "" FORCE)
set(LLVM_ENABLE_RTTI ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLVM_BUILD_LLVM_DYLIB OFF CACHE BOOL "" FORCE)
set(LLVM_LINK_LLVM_DYLIB OFF CACHE BOOL "" FORCE)

# Speed-ups / sanity
set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_DOCS OFF CACHE BOOL "" FORCE)
set(MLIR_ENABLE_BINDINGS_PYTHON OFF CACHE BOOL "" FORCE)

# Add LLVM as a subproject (monorepo top-level is at llvm/)
add_subdirectory("${LLVM_SOURCE_DIR}" "${LLVM_BINARY_DIR}" EXCLUDE_FROM_ALL)

# Point CMake package lookups to the build-tree configs
set(LLVM_DIR "${LLVM_BINARY_DIR}/lib/cmake/llvm" CACHE PATH "" FORCE)
set(MLIR_DIR "${LLVM_BINARY_DIR}/lib/cmake/mlir" CACHE PATH "" FORCE)
