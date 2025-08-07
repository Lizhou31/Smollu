# Virtual Machine components

# Add Smollu VM library
add_library(smollu_vm STATIC src/components/vm/smollu_vm.c)

target_include_directories(smollu_vm PUBLIC
    src/components/vm
)