# Smollu MLIR Compiler
cmake_minimum_required(VERSION 3.15)

# Enable C++
enable_language(CXX)
set(CMAKE_CXX_STANDARD 17)

# Match MLIR compilation flags (disable RTTI to match LLVM/MLIR)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")

# Find LLVM and MLIR
find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")

# Set up LLVM/MLIR paths
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

# Skip HandleLLVMOptions to avoid version check issues
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/)

# Definitions
add_definitions(${LLVM_DEFINITIONS})

# TableGen for dialect
set(LLVM_TARGET_DEFINITIONS include/Smollu/SmolluOps.td)
mlir_tablegen(SmolluOps.h.inc -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(SmolluOps.cpp.inc -gen-op-defs -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(SmolluDialect.h.inc -gen-dialect-decls -dialect=smollu -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(SmolluDialect.cpp.inc -gen-dialect-defs -dialect=smollu -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_public_tablegen_target(SmolluOpsIncGen)

# Dialect library
add_library(SmolluDialect
    lib/Dialect/SmolluDialect.cpp
)
target_link_libraries(SmolluDialect
    MLIRDialect
    MLIRIR
    MLIRSupport
)
target_include_directories(SmolluDialect PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(SmolluDialect SmolluOpsIncGen)

# Parser library
add_library(SmolluParser
    lib/Parser/Parser.cpp
    lib/Parser/ASTParser.cpp
    lib/Parser/MLIRGenerator.cpp
    lib/Parser/SmolluParser.cpp
)
target_link_libraries(SmolluParser
    SmolluDialect
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    smollu_lexer
)

# CodeGen library
add_library(SmolluCodeGen
    lib/CodeGen/BytecodeEmitter.cpp
    lib/CodeGen/ASMEmitter.cpp
)
target_link_libraries(SmolluCodeGen
    SmolluDialect
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
)

# Compiler executable
add_executable(smollu-mlir-compiler
    src/smollu-mlir-compiler.cpp
)
target_link_libraries(smollu-mlir-compiler
    SmolluParser
    SmolluCodeGen
    SmolluDialect
    MLIRDialect
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    LLVMSupport
)