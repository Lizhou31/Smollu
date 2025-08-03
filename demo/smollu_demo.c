#include "smollu_vm.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "demo_bytecode.h"
#define SMOLLU_INTERPRETER_MAIN

/* ────────────────────────────────────────────────────────────────────────── */
/*  Sample native bindings                                                    */
/* ────────────────────────────────────────────────────────────────────────── */

/* Native #0  : print (variadic) */
static Value nat_print(Value *args, uint8_t argc) {
    for (uint8_t i = 0; i < argc; ++i) {
        Value v = args[i];
        switch (v.type) {
            case VAL_NIL:   printf("nil"); break;
            case VAL_BOOL:  printf(v.as.boolean ? "true" : "false"); break;
            case VAL_INT:   printf("%d", v.as.i); break;
            case VAL_FLOAT: printf("%f", v.as.f); break;
        }
        if (i + 1 < argc) printf(" ");
    }
    printf("\n");
    return value_make_nil();
}

/* Native #1  : random int in range [0, arg0) */
static Value nat_rand(Value *args, uint8_t argc) {
    (void)argc;
    if (argc == 0 || args[0].type != VAL_INT) return value_from_int(0);
    int limit = args[0].as.i;
    return value_from_int(rand() % (limit ? limit : 1));
}

/* Device-specific native table */
static const NativeFn device_native_table[] = { nat_print, nat_rand };

/* ────────────────────────────────────────────────────────────────────────── */
/*  Public helper                                                             */
/* ────────────────────────────────────────────────────────────────────────── */

int smollu_interpreter_run(void) {
    SmolluVM vm;
    smollu_vm_init(&vm);

    /* Read header and register native functions according to the image */
    smollu_vm_prepare(&vm, demo_header_and_table, device_native_table);

    /* Load the code section */
    smollu_vm_load(&vm, demo_code, demo_code_len);

    int rc = smollu_vm_run(&vm);
    if (rc == 0) {
        printf("[VM] Program terminated normally.\n");
    } else {
        printf("[VM] Program terminated with error code %d.\n", rc);
    }

    smollu_vm_destroy(&vm);
    return rc;
}

/* ────────────────────────────────────────────────────────────────────────── */
/*  Stand-alone driver (optional)                                             */
/* ────────────────────────────────────────────────────────────────────────── */

#ifdef SMOLLU_INTERPRETER_MAIN
int main(int argc, char **argv) {
    int rc = smollu_interpreter_run();
    return rc;
}
#endif /* SMOLLU_INTERPRETER_MAIN */
