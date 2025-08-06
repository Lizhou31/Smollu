cmake_minimum_required(VERSION 3.15)
project(smollu_vm C)

set(CMAKE_C_STANDARD 99)

# ---------------------------------------------------------------------------
#  Source files
# ---------------------------------------------------------------------------
set(SMOLLU_LEXER_SRC
    src/components/compiler/smollu_lexer.c
)

add_library(smollu_lexer STATIC ${SMOLLU_LEXER_SRC})
target_include_directories(smollu_lexer PUBLIC
    src/components/compiler
)

set(SMOLLU_PARSER_SRC
    src/components/compiler/smollu_parser.c
)

add_library(smollu_parser STATIC ${SMOLLU_PARSER_SRC})
target_include_directories(smollu_parser PUBLIC
    src/components/compiler
)
# Link lexer into parser
target_link_libraries(smollu_parser PUBLIC smollu_lexer)

set(SMOLLU_CODEGEN_SRC
    src/components/compiler/smollu_bytecode_codegen.c
)

add_library(smollu_codegen STATIC ${SMOLLU_CODEGEN_SRC})
# codegen uses compiler headers and VM native tables header resides there too
target_include_directories(smollu_codegen PUBLIC
    src/components/compiler
)

target_link_libraries(smollu_codegen PUBLIC smollu_parser smollu_lexer)

add_executable(smollu_compiler
    src/components/compiler/smollu_compiler.c
)
target_include_directories(smollu_compiler PUBLIC
    src/components/compiler
)
target_link_libraries(smollu_compiler PUBLIC smollu_codegen)

# Add Smollu VM library
add_library(smollu_vm STATIC src/components/vm/smollu_vm.c)

target_include_directories(smollu_vm PUBLIC
    src/components/vm
)

# ---------------------------------------------------------------------------
#  Tests (Criterion)
# ---------------------------------------------------------------------------
include(CTest)
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CRITERION REQUIRED criterion)

    add_executable(test_smollu_lexer
        test/test_smollu_lexer.c
    )

    target_include_directories(test_smollu_lexer PRIVATE
        ${CRITERION_INCLUDE_DIRS}
        src/components/compiler
    )
    target_link_libraries(test_smollu_lexer
        smollu_lexer
        ${CRITERION_LIBRARIES}
    )
    target_compile_options(test_smollu_lexer PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_lexer PRIVATE ${CRITERION_LIBRARY_DIRS})

    add_test(NAME lexer_tests COMMAND test_smollu_lexer --verbose)

    add_executable(test_smollu_parser
        test/test_smollu_parser.c
    )

    target_include_directories(test_smollu_parser PRIVATE
        ${CRITERION_INCLUDE_DIRS}
        src/components/compiler
    )

    target_link_libraries(test_smollu_parser
        smollu_parser
        ${CRITERION_LIBRARIES}
    )

    target_compile_options(test_smollu_parser PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_parser PRIVATE ${CRITERION_LIBRARY_DIRS})

    add_test(NAME parser_tests COMMAND test_smollu_parser --verbose)

    # -------------------- VM tests --------------------
    add_executable(test_smollu_vm
        test/test_smollu_vm.c
    )

    target_include_directories(test_smollu_vm PRIVATE
        ${CRITERION_INCLUDE_DIRS}
        src/components/vm
    )

    target_link_libraries(test_smollu_vm
        smollu_vm
        ${CRITERION_LIBRARIES}
    )

    target_compile_options(test_smollu_vm PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_vm PRIVATE ${CRITERION_LIBRARY_DIRS})

    add_test(NAME vm_tests COMMAND test_smollu_vm --verbose)

    # -------------------- Bytecode Codegen tests --------------------
    add_executable(test_smollu_bytecode_codegen
        test/test_smollu_bytecode_codegen.c
    )

    target_include_directories(test_smollu_bytecode_codegen PRIVATE
        ${CRITERION_INCLUDE_DIRS}
        src/components/compiler
        src/components/vm
    )

    target_link_libraries(test_smollu_bytecode_codegen
        smollu_codegen
        ${CRITERION_LIBRARIES}
    )

    target_compile_options(test_smollu_bytecode_codegen PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_bytecode_codegen PRIVATE ${CRITERION_LIBRARY_DIRS})

    add_test(NAME bytecode_codegen_tests COMMAND test_smollu_bytecode_codegen --verbose)
endif()

# ---------------------------------------------------------------------------
#  Demo executable
# ---------------------------------------------------------------------------
add_executable(smollu_demo
    demo/smollu_demo.c
)
target_include_directories(smollu_demo PUBLIC
    src/components/vm
    src/components/compiler
    demo
)
target_link_libraries(smollu_demo PUBLIC smollu_vm)

# ---------------------------------------------------------------------------
#  Usage
# ---------------------------------------------------------------------------
# mkdir build && cd build
# cmake .. -DBUILD_TESTS=ON
# cmake --build .
# ctest --verbose 