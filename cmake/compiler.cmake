# Compiler components (lexer, parser, codegen, compiler executable)

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