# Demo executable configuration

add_executable(smollu_demo
    "demo/Simple demo/smollu_demo.c"
)
target_include_directories(smollu_demo PUBLIC
    vm
    compiler
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

file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/demo/LED\ Matrix\ demo)
configure_file(
    ${CMAKE_SOURCE_DIR}/demo/LED\ Matrix\ demo/basic_led_demo.smol
    ${CMAKE_BINARY_DIR}/demo/LED\ Matrix\ demo/basic_led_demo.smol
    COPYONLY
)
configure_file(
    ${CMAKE_SOURCE_DIR}/demo/LED\ Matrix\ demo/animation_demo.smol
    ${CMAKE_BINARY_DIR}/demo/LED\ Matrix\ demo/animation_demo.smol
    COPYONLY
)

# Copy executables to demo directory after build
add_custom_command(TARGET smollu_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smollu_demo> ${CMAKE_BINARY_DIR}/demo/Simple\ demo/
    COMMENT "Copying smollu_demo to demo directory"
)

# Create a custom target to copy compiler after it's built
add_custom_target(copy_compiler_to_demo ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smollu_compiler> ${CMAKE_BINARY_DIR}/demo/Simple\ demo/
    COMMENT "Copying smollu_compiler to demo directory"
    DEPENDS smollu_compiler
)

add_custom_target(copy_compiler_to_led_matrix_demo ALL
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:smollu_compiler> ${CMAKE_BINARY_DIR}/demo/LED\ Matrix\ demo/
    COMMENT "Copying smollu_compiler to demo directory"
    DEPENDS smollu_compiler
)