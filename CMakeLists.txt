cmake_minimum_required(VERSION 3.15)
project(smollu_vm C)

set(CMAKE_C_STANDARD 99)

# Add cmake modules to path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---------------------------------------------------------------------------
#  Core Components
# ---------------------------------------------------------------------------
include(vm)
include(compiler)

# ---------------------------------------------------------------------------
#  Optional Components
# ---------------------------------------------------------------------------
include(emulator)
include(tests)
include(demo)

# ---------------------------------------------------------------------------
#  Usage
# ---------------------------------------------------------------------------
# mkdir build && cd build
# cmake .. -DBUILD_TESTS=ON
# cmake --build .
# ctest --verbose 