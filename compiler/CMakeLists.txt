# Smollu Compiler Component
cmake_minimum_required(VERSION 3.15)

# Lexer Library
add_library(smollu_lexer STATIC smollu_lexer.c)
target_include_directories(smollu_lexer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Parser Library
add_library(smollu_parser STATIC smollu_parser.c)
target_include_directories(smollu_parser PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(smollu_parser PUBLIC smollu_lexer)

# Bytecode Generator Library
add_library(smollu_codegen STATIC smollu_bytecode_codegen.c)
target_include_directories(smollu_codegen PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(smollu_codegen PUBLIC smollu_parser smollu_lexer)

# Compiler Executable
add_executable(smollu_compiler smollu_compiler.c)
target_include_directories(smollu_compiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(smollu_compiler PUBLIC smollu_codegen)

# Compiler Tests
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(CRITERION REQUIRED criterion)

    # Lexer tests
    add_executable(test_smollu_lexer
        test/test_smollu_lexer.c
    )

    target_include_directories(test_smollu_lexer PRIVATE
        ${CRITERION_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_link_libraries(test_smollu_lexer
        smollu_lexer
        ${CRITERION_LIBRARIES}
    )
    target_compile_options(test_smollu_lexer PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_lexer PRIVATE ${CRITERION_LIBRARY_DIRS})

    # Parser tests
    add_executable(test_smollu_parser
        test/test_smollu_parser.c
    )

    target_include_directories(test_smollu_parser PRIVATE
        ${CRITERION_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    target_link_libraries(test_smollu_parser
        smollu_parser
        ${CRITERION_LIBRARIES}
    )

    target_compile_options(test_smollu_parser PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_parser PRIVATE ${CRITERION_LIBRARY_DIRS})

    # Bytecode Codegen tests
    add_executable(test_smollu_bytecode_codegen
        test/test_smollu_bytecode_codegen.c
    )

    target_include_directories(test_smollu_bytecode_codegen PRIVATE
        ${CRITERION_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/vm  # Need VM headers for bytecode tests
    )

    target_link_libraries(test_smollu_bytecode_codegen
        smollu_codegen
        ${CRITERION_LIBRARIES}
    )

    target_compile_options(test_smollu_bytecode_codegen PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_bytecode_codegen PRIVATE ${CRITERION_LIBRARY_DIRS})

    # Tests are registered at the top level in cmake/tests.cmake
endif()