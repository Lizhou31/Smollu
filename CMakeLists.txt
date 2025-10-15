cmake_minimum_required(VERSION 3.15)
project(smollu_vm C)

set(CMAKE_C_STANDARD 99)

# Add cmake modules to path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
#  LLVM/MLIR Submodule Configuration
# ---------------------------------------------------------------------------
option(SMOLLU_USE_LLVM_SUBMODULE "Build LLVM/MLIR from submodule" ON)

# ---------------------------------------------------------------------------
#  Core Components
# ---------------------------------------------------------------------------
include(vm)
include(compiler)

# ---------------------------------------------------------------------------
#  Optional Components
# ---------------------------------------------------------------------------
# Build LLVM/MLIR from submodule if enabled and available
if(SMOLLU_USE_LLVM_SUBMODULE AND EXISTS "${CMAKE_SOURCE_DIR}/external/llvm-project/llvm/CMakeLists.txt")
  include(llvm_from_submodule)
endif()

include(emulator)
include(tests)
include(demo)
include(compiler_mlir)

# ---------------------------------------------------------------------------
#  Usage
# ---------------------------------------------------------------------------
# mkdir build && cd build
# cmake .. -DBUILD_TESTS=ON
# cmake --build .
# ctest --verbose 