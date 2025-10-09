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

# TableGen for Smol dialect (high-level, language semantics)
set(LLVM_TARGET_DEFINITIONS include/Smollu/SmolOps.td)
mlir_tablegen(SmolOps.h.inc -gen-op-decls -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(SmolOps.cpp.inc -gen-op-defs -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(SmolDialect.h.inc -gen-dialect-decls -dialect=smol -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
mlir_tablegen(SmolDialect.cpp.inc -gen-dialect-defs -dialect=smol -I ${CMAKE_CURRENT_SOURCE_DIR}/include)
add_public_tablegen_target(SmolOpsIncGen)

# Smol dialect library (high-level)
add_library(SmolDialect
    lib/Dialect/SmolDialect.cpp
)
target_link_libraries(SmolDialect
    MLIRDialect
    MLIRIR
    MLIRSupport
    MLIRControlFlowInterfaces
    MLIRSideEffectInterfaces
)
target_include_directories(SmolDialect PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
add_dependencies(SmolDialect SmolOpsIncGen)

# C++ Lexer library (with MLIR location tracking)
add_library(SmolluLexer
    lib/Lexer/Lexer.cpp
)
target_link_libraries(SmolluLexer
    MLIRSupport
)
target_include_directories(SmolluLexer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Parser library
add_library(SmolluParser
    lib/Parser/ASTParser.cpp
    lib/Parser/SmolMLIRGen.cpp
    lib/Parser/SmolluParser.cpp
)
target_link_libraries(SmolluParser
    SmolluLexer
    SmolDialect
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
)

# Pass library for transformations
add_library(SmolPasses
    lib/Pass/PromoteNumerics.cpp
    lib/Pass/ScopeVerification.cpp
    lib/Pass/DeadCodeElimination.cpp
)
target_link_libraries(SmolPasses
    SmolDialect
    MLIRIR
    MLIRPass
    MLIRSupport
    MLIRTransforms
)
target_include_directories(SmolPasses PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

# CodeGen library (temporarily disabled - will be refactored for Smol dialect)
add_library(SmolluCodeGen
    lib/CodeGen/BytecodeEmitter.cpp
    lib/CodeGen/ASMEmitter.cpp
)
target_link_libraries(SmolluCodeGen
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
    SmolDialect
    SmolPasses
    MLIRDialect
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect
    MLIRTransforms
    LLVMSupport
)