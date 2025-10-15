# MLIR-based Smollu compiler
# Optional component - requires LLVM/MLIR to be installed

option(BUILD_MLIR_COMPILER "Build MLIR-based Smollu compiler" ON)

if(BUILD_MLIR_COMPILER)
    # When using submodule, targets are available directly; no find_package needed
    if(SMOLLU_USE_LLVM_SUBMODULE AND LLVM_DIR)
        message(STATUS "Building MLIR-based Smollu compiler (from submodule)")
        add_subdirectory(compiler_mlir)
    else()
        # Check if LLVM/MLIR is available from system installation
        find_package(LLVM QUIET CONFIG)
        find_package(MLIR QUIET CONFIG)

        if(LLVM_FOUND AND MLIR_FOUND)
            message(STATUS "Building MLIR-based Smollu compiler (system install)")
            add_subdirectory(compiler_mlir)
        else()
            message(WARNING "LLVM/MLIR not found. Skipping MLIR compiler build.")
            message(STATUS "To install LLVM/MLIR:")
            message(STATUS "  macOS: brew install llvm")
            message(STATUS "  Ubuntu: sudo apt install llvm-dev mlir-tools libmlir-dev")
        endif()
    endif()
endif()