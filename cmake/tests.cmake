# Unit tests configuration

include(CTest)
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_STREAMING_TEST "Build streaming VM test" ON)

# Register tests at the top level to ensure CTest finds them
if(BUILD_TESTS)
    # VM tests
    add_test(NAME vm_tests COMMAND vm/test_smollu_vm --verbose)

    # Compiler tests
    add_test(NAME lexer_tests COMMAND compiler/test_smollu_lexer --verbose)
    add_test(NAME parser_tests COMMAND compiler/test_smollu_parser --verbose)
    add_test(NAME bytecode_codegen_tests COMMAND compiler/test_smollu_bytecode_codegen --verbose)
endif()

# Streaming VM test (requires bytecode file)
if(BUILD_STREAMING_TEST)
    add_test(NAME streaming_vm_test
             COMMAND vm/test_streaming_vm ${CMAKE_BINARY_DIR}/demo/Simple\ demo/demo.smolbc
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    # This test depends on demo bytecode being compiled
    # Note: Using FIXTURES_REQUIRED would be better, but DEPENDS works for build-time dependencies
    set_tests_properties(streaming_vm_test PROPERTIES
        REQUIRED_FILES ${CMAKE_BINARY_DIR}/demo/Simple\ demo/demo.smolbc
    )
endif()