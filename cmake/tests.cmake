# Unit tests configuration

include(CTest)
option(BUILD_TESTS "Build unit tests" ON)

# Register tests at the top level to ensure CTest finds them
if(BUILD_TESTS)
    # VM tests
    add_test(NAME vm_tests COMMAND vm/test_smollu_vm --verbose)
    
    # Compiler tests  
    add_test(NAME lexer_tests COMMAND compiler/test_smollu_lexer --verbose)
    add_test(NAME parser_tests COMMAND compiler/test_smollu_parser --verbose)
    add_test(NAME bytecode_codegen_tests COMMAND compiler/test_smollu_bytecode_codegen --verbose)
endif()