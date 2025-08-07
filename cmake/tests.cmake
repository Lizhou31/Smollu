# Unit tests configuration

include(CTest)
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
        src/components/compiler
    )
    target_link_libraries(test_smollu_lexer
        smollu_lexer
        ${CRITERION_LIBRARIES}
    )
    target_compile_options(test_smollu_lexer PRIVATE ${CRITERION_CFLAGS_OTHER})
    target_link_directories(test_smollu_lexer PRIVATE ${CRITERION_LIBRARY_DIRS})

    add_test(NAME lexer_tests COMMAND test_smollu_lexer --verbose)

    # Parser tests
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

    # VM tests
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

    # Bytecode Codegen tests
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