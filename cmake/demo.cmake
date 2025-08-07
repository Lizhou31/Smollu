# Demo executable configuration

add_executable(smollu_demo
    "demo/Simple demo/smollu_demo.c"
)
target_include_directories(smollu_demo PUBLIC
    src/components/vm
    src/components/compiler
    "demo/Simple demo"
)
target_link_libraries(smollu_demo PUBLIC smollu_vm)

# Create demo directory in build folder and copy demo.smol
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/demo/Simple\ demo)
configure_file(
    ${CMAKE_SOURCE_DIR}/demo/Simple\ demo/demo.smol
    ${CMAKE_BINARY_DIR}/demo/Simple\ demo/demo.smol
    COPYONLY
)

# Copy executables to demo directory after build
add_custom_command(TARGET smollu_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smollu_demo> ${CMAKE_BINARY_DIR}/demo/Simple\ demo/
    COMMENT "Copying smollu_demo to demo directory"
)

add_custom_command(TARGET smollu_compiler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smollu_compiler> ${CMAKE_BINARY_DIR}/demo/Simple\ demo/
    COMMENT "Copying smollu_compiler to demo directory"
)