# MLIR-based Smollu compiler
# Optional component - requires LLVM/MLIR to be installed

option(BUILD_MLIR_COMPILER "Build MLIR-based Smollu compiler" ON)

if(BUILD_MLIR_COMPILER)
    # Check if LLVM/MLIR is available
    find_package(LLVM QUIET CONFIG)
    find_package(MLIR QUIET CONFIG)

    if(LLVM_FOUND AND MLIR_FOUND)
        message(STATUS "Building MLIR-based Smollu compiler")
        add_subdirectory(compiler_mlir)
    else()
        message(WARNING "LLVM/MLIR not found. Skipping MLIR compiler build.")
        message(STATUS "To install LLVM/MLIR:")
        message(STATUS "  macOS: brew install llvm")
        message(STATUS "  Ubuntu: sudo apt install llvm-dev mlir-tools libmlir-dev")
    endif()
endif()